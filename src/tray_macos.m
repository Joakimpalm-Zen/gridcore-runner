// macOS menu-bar backend: NSStatusItem + NSMenu. Rendering only — every
// menu is built by tray_menu_build() at open time and clicks go straight
// back through tray_menu_act(). Accessory activation policy: no Dock icon,
// no menu bar of our own.
#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include "tray.h"

#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;

#define TRAY_MAX_ITEMS 128

static NSStatusItem *g_status;

// ---------------------------------------------------------------- the icon
// One rounded-square core with a signal motif around it, drawn in code as a
// template image so macOS recolors it for light and dark menu bars. Three
// states, matching the three the core can actually distinguish:
//
//   IDLE     hollow core, two sweeps      nothing loaded
//   LOADED   solid core, two sweeps       model resident, waiting
//   RUNNING  solid core, full broken ring inference in flight
//
// The ring is segmented rather than continuous because a menu-bar template
// image cannot animate: four gaps read as motion where a closed circle would
// read as a static badge. Everything is drawn from the centre of an 18×18 box
// so the three glyphs share an optical weight and the icon does not appear to
// shift when the state changes.
static NSImage *core_icon_px(tray_icon_state st, CGFloat px) {
    const CGFloat s = px / 18.0;   // geometry below is authored in 18-pt units
    NSImage *img = [NSImage imageWithSize:NSMakeSize(px, px)
                                  flipped:NO
                           drawingHandler:^BOOL(NSRect rect) {
        (void)rect;
        [[NSColor blackColor] setFill];
        [[NSColor blackColor] setStroke];
        const CGFloat cx = 9.0 * s, cy = 9.0 * s;

        NSPoint c = NSMakePoint(cx, cy);
        void (^sweep)(CGFloat, CGFloat, CGFloat, CGFloat) =
            ^(CGFloat r, CGFloat mid, CGFloat half, CGFloat w) {
            NSBezierPath *p = [NSBezierPath bezierPath];
            p.lineWidth = w;
            p.lineCapStyle = NSLineCapStyleRound;
            [p appendBezierPathWithArcWithCenter:c
                                          radius:r
                                      startAngle:mid - half
                                        endAngle:mid + half];
            [p stroke];
        };

        // the core
        const CGFloat side = 7.2 * s;
        if (st == TRAY_ICON_IDLE) {
            // inset by half the line width so the hollow core keeps the solid
            // one's outer edge instead of growing outward by half a stroke
            const CGFloat w = 1.3 * s;
            NSBezierPath *core = [NSBezierPath bezierPathWithRoundedRect:
                NSMakeRect(cx - side / 2 + w / 2, cy - side / 2 + w / 2,
                           side - w, side - w)
                                                                xRadius:1.8 * s
                                                                yRadius:1.8 * s];
            core.lineWidth = w;
            [core stroke];
        } else {
            [[NSBezierPath bezierPathWithRoundedRect:
                NSMakeRect(cx - side / 2, cy - side / 2, side, side)
                                             xRadius:2.2 * s
                                             yRadius:2.2 * s] fill];
        }

        if (st == TRAY_ICON_RUNNING) {
            for (int i = 0; i < 4; i++)
                sweep(6.5 * s, 45.0 + i * 90.0, 32.0, 1.5 * s);
        } else {
            // two opposing sweeps: upper-right and lower-left
            sweep(6.3 * s, 45.0, 30.0, 1.3 * s);
            sweep(6.3 * s, 225.0, 30.0, 1.3 * s);
        }
        return YES;
    }];
    [img setTemplate:YES];
    return img;
}

static NSImage *core_icon(tray_icon_state st) { return core_icon_px(st, 18.0); }

bool tray_platform_icon_dump(const char *dir, int px) {
    @autoreleasepool {
        const char *names[] = { "idle", "loaded", "running" };
        for (int i = 0; i < 3; i++) {
            // template images carry no colour, so paint the glyph onto an
            // opaque white ground — otherwise the PNG is black-on-transparent
            // and unreviewable in most viewers
            NSImage *glyph = core_icon_px((tray_icon_state)i, px);
            NSBitmapImageRep *rep = [[NSBitmapImageRep alloc]
                initWithBitmapDataPlanes:NULL pixelsWide:px pixelsHigh:px
                            bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES
                                 isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace
                              bytesPerRow:0 bitsPerPixel:0];
            [NSGraphicsContext saveGraphicsState];
            NSGraphicsContext.currentContext =
                [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
            [[NSColor whiteColor] setFill];
            NSRectFill(NSMakeRect(0, 0, px, px));
            [glyph drawInRect:NSMakeRect(0, 0, px, px)];
            [NSGraphicsContext restoreGraphicsState];

            NSData *png = [rep representationUsingType:NSBitmapImageFileTypePNG
                                            properties:@{}];
            char path[1200];
            snprintf(path, sizeof path, "%s/tray-%s.png", dir, names[i]);
            if (![png writeToFile:[NSString stringWithUTF8String:path]
                       atomically:YES])
                return false;
            printf("wrote %s\n", path);
        }
    }
    return true;
}

// ------------------------------------------------------------- menu bridge

@interface TrayDelegate : NSObject <NSMenuDelegate>
- (void)clicked:(NSMenuItem *)sender;
@end

@implementation TrayDelegate

- (void)menuNeedsUpdate:(NSMenu *)menu {
    [menu removeAllItems];
    static tray_item items[TRAY_MAX_ITEMS];
    int n = tray_menu_build(items, TRAY_MAX_ITEMS);

    NSMenu *cur = menu;
    NSMenuItem *last = nil;
    for (int i = 0; i < n; i++) {
        tray_item *t = &items[i];
        switch (t->kind) {
        case TRAY_K_SEP:
            [cur addItem:[NSMenuItem separatorItem]];
            break;
        case TRAY_K_SUB_BEGIN: {
            NSMenu *sub = [[NSMenu alloc] init];
            sub.autoenablesItems = NO;
            if (last) {
                // AppKit refuses to open a submenu hanging off a disabled
                // item — the parent row must be enabled (it has no action,
                // so clicking it still does nothing)
                last.enabled = YES;
                [cur setSubmenu:sub forItem:last];
            }
            cur = sub;
            break;
        }
        case TRAY_K_SUB_END:
            cur = menu;
            break;
        default: {
            NSMenuItem *mi = [[NSMenuItem alloc]
                initWithTitle:[NSString stringWithUTF8String:t->label]
                       action:(t->kind == TRAY_K_LABEL ? nil : @selector(clicked:))
                keyEquivalent:@""];
            mi.target = (t->kind == TRAY_K_LABEL) ? nil : self;
            mi.enabled = (t->kind != TRAY_K_LABEL);
            mi.tag = i;
            mi.representedObject = @[ @(t->action), @(t->arg) ];
            if (t->kind == TRAY_K_CHECK)
                mi.state = t->checked ? NSControlStateValueOn : NSControlStateValueOff;
            [cur addItem:mi];
            last = mi;
            break;
        }
        }
    }
    g_status.button.image = core_icon(tray_icon());
}

- (void)clicked:(NSMenuItem *)sender {
    NSArray *ra = sender.representedObject;
    int action = [ra[0] intValue];
    long arg = [ra[1] longValue];

    if (action == TRAY_ACT_PICK_MODEL) {
        NSOpenPanel *panel = [NSOpenPanel openPanel];
        panel.canChooseFiles = YES;
        panel.canChooseDirectories = NO;
        panel.allowsMultipleSelection = NO;
        UTType *gguf = [UTType typeWithFilenameExtension:@"gguf"];
        if (gguf) panel.allowedContentTypes = @[ gguf ];
        panel.message = @"Choose a GGUF model for the desktop-managed runner";
        [NSApp activateIgnoringOtherApps:YES];
        if ([panel runModal] == NSModalResponseOK && panel.URL)
            tray_menu_act(action, 0, panel.URL.path.UTF8String);
        else
            tray_menu_act(action, 0, NULL);
    } else {
        tray_menu_act(action, arg, NULL);
    }

    if (tray_should_quit())
        [NSApp stop:nil];
    else
        g_status.button.image = core_icon(tray_icon());
}

- (void)tick:(NSTimer *)timer {
    if (tray_should_quit()) { [NSApp stop:nil]; return; }
    g_status.button.image = core_icon(tray_icon());
}

@end

// -------------------------------------------------------------- autostart
// LaunchAgent at ~/Library/LaunchAgents/ai.xyntetik.runner.tray.plist.
// Presence of the file IS the state; no launchctl bookkeeping in v1.

static void agent_path(char *out, size_t cap) {
    const char *home = getenv("HOME");
    snprintf(out, cap, "%s/Library/LaunchAgents/ai.xyntetik.runner.tray.plist",
             home ? home : ".");
}

static void unload_old_agent(void) {
    char *argv[] = {
        (char *)"launchctl", (char *)"remove",
        (char *)"ai.gridcore.runner.tray", NULL,
    };
    posix_spawn_file_actions_t fa;
    if (posix_spawn_file_actions_init(&fa) != 0) return;
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);

    pid_t pid;
    int rc = posix_spawnp(&pid, argv[0], &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return;

    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {}
}

// One-time migration of the pre-rename agent (Gridcore -> Xyntetik). The old
// label must be unloaded before the new one registers, or two trays can end
// up installed; autostart state is preserved by re-creating the agent under
// the new label.
static void migrate_old_agent(void) {
    static bool done = false;
    if (done) return;
    done = true;
    const char *home = getenv("HOME");
    char op[1024];
    snprintf(op, sizeof op,
             "%s/Library/LaunchAgents/ai.gridcore.runner.tray.plist",
             home ? home : ".");
    FILE *f = fopen(op, "rb");
    if (!f) return;
    fclose(f);
    unload_old_agent();
    remove(op);
    tray_platform_autostart_set(true);
}

bool tray_platform_autostart_get(void) {
    migrate_old_agent();
    char p[1024];
    agent_path(p, sizeof p);
    FILE *f = fopen(p, "rb");
    if (f) fclose(f);
    return f != NULL;
}

bool tray_platform_autostart_set(bool on) {
    char p[1024];
    agent_path(p, sizeof p);
    if (!on) return remove(p) == 0;

    char exe[1200];
    uint32_t sz = sizeof exe;
    extern int _NSGetExecutablePath(char *, uint32_t *);
    if (_NSGetExecutablePath(exe, &sz) != 0) return false;

    FILE *f = fopen(p, "wb");
    if (!f) return false;
    fprintf(f,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\""
        " \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        "<plist version=\"1.0\"><dict>\n"
        "  <key>Label</key><string>ai.xyntetik.runner.tray</string>\n"
        "  <key>ProgramArguments</key><array>\n"
        "    <string>%s</string>\n"
        "    <string>--tray</string>\n"
        "  </array>\n"
        "  <key>RunAtLoad</key><true/>\n"
        "</dict></plist>\n",
        exe);
    return fclose(f) == 0;
}

// -------------------------------------------------------------- main loop

int tray_platform_run(void) {
    migrate_old_agent();
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

        TrayDelegate *del = [[TrayDelegate alloc] init];
        g_status = [[NSStatusBar systemStatusBar]
            statusItemWithLength:NSSquareStatusItemLength];
        g_status.button.image = core_icon(tray_icon());
        g_status.button.toolTip = @"xyntetik-runner";

        NSMenu *menu = [[NSMenu alloc] init];
        menu.autoenablesItems = NO;
        menu.delegate = del;
        g_status.menu = menu;

        // badge refresh while the menu is closed (menuNeedsUpdate only
        // fires on open)
        [NSTimer scheduledTimerWithTimeInterval:5.0
                                         target:del
                                       selector:@selector(tick:)
                                       userInfo:nil
                                        repeats:YES];

        [NSApp run];
        [[NSStatusBar systemStatusBar] removeStatusItem:g_status];
    }
    return 0;
}
