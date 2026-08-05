// macOS menu-bar backend: NSStatusItem + NSMenu. Rendering only — every
// menu is built by tray_menu_build() at open time and clicks go straight
// back through tray_menu_act(). Accessory activation policy: no Dock icon,
// no menu bar of our own.
#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include "tray.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAY_MAX_ITEMS 128

static NSStatusItem *g_status;

// ---------------------------------------------------------------- the icon
// Code-drawn 3×3 grid glyph, template image so macOS recolors it for
// light/dark menu bars. `filled` adds the running-dot in the center cell.
static NSImage *grid_icon(bool filled) {
    NSImage *img = [NSImage imageWithSize:NSMakeSize(18, 18)
                                  flipped:NO
                           drawingHandler:^BOOL(NSRect rect) {
        [[NSColor blackColor] setFill];
        CGFloat cell = 4.0, gap = 1.5, x0 = 1.5, y0 = 1.5;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) {
                NSRect cr = NSMakeRect(x0 + c * (cell + gap),
                                       y0 + r * (cell + gap), cell, cell);
                NSBezierPath *p =
                    [NSBezierPath bezierPathWithRoundedRect:cr xRadius:1 yRadius:1];
                if (r == 1 && c == 1 && filled) {
                    [p fill];
                } else {
                    p.lineWidth = 1.0;
                    [p stroke];
                }
            }
        return YES;
    }];
    [img setTemplate:YES];
    return img;
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
            if (last) [cur setSubmenu:sub forItem:last];
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
    g_status.button.image = grid_icon(tray_any_running());
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
        g_status.button.image = grid_icon(tray_any_running());
}

- (void)tick:(NSTimer *)timer {
    if (tray_should_quit()) { [NSApp stop:nil]; return; }
    g_status.button.image = grid_icon(tray_any_running());
}

@end

// -------------------------------------------------------------- autostart
// LaunchAgent at ~/Library/LaunchAgents/ai.gridcore.runner.tray.plist.
// Presence of the file IS the state; no launchctl bookkeeping in v1.

static void agent_path(char *out, size_t cap) {
    const char *home = getenv("HOME");
    snprintf(out, cap, "%s/Library/LaunchAgents/ai.gridcore.runner.tray.plist",
             home ? home : ".");
}

bool tray_platform_autostart_get(void) {
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
        "  <key>Label</key><string>ai.gridcore.runner.tray</string>\n"
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
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];

        TrayDelegate *del = [[TrayDelegate alloc] init];
        g_status = [[NSStatusBar systemStatusBar]
            statusItemWithLength:NSSquareStatusItemLength];
        g_status.button.image = grid_icon(tray_any_running());
        g_status.button.toolTip = @"gridcore-runner";

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
