// Windows tray backend: Shell_NotifyIcon + a hidden message-only window.
// The runner stays a console-subsystem binary (no -mwindows — that would
// break every CLI mode); when launched with --tray from Explorer the
// console is released with FreeConsole so no black window lingers.
#ifdef _WIN32
#include "tray.h"

#include <windows.h>
#include <shellapi.h>
#include <commdlg.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAY_MAX_ITEMS 128
#define WM_TRAY_CALLBACK (WM_APP + 1)
#define TRAY_ICON_ID 1
#define MENU_ID_BASE 1000

static NOTIFYICONDATAA g_nid;
static HWND g_hwnd;
static tray_item g_items[TRAY_MAX_ITEMS];
static int g_nitems;

// ---------------------------------------------------------------- the icon
// Same three-state glyph as macOS, painted white-on-black into a 16×16 HICON:
// a rounded-square core, hollow when nothing is loaded, with two opposing
// sweeps — or a four-segment ring while inference is in flight.
//
// GDI's Arc() always travels counter-clockwise from the start radial to the
// end radial, and device y grows DOWNWARD. Negating the sine puts the glyph in
// the same visual orientation as the macOS one; because both endpoints are
// negated together, the sweep direction is preserved and Arc still traces the
// short way between them rather than the complement.
static void arc_seg(HDC dc, double cx, double cy, double r,
                    double mid_deg, double half_deg) {
    const double D2R = 3.14159265358979323846 / 180.0;
    double a0 = (mid_deg - half_deg) * D2R, a1 = (mid_deg + half_deg) * D2R;
    Arc(dc, (int)lround(cx - r), (int)lround(cy - r),
            (int)lround(cx + r), (int)lround(cy + r),
            (int)lround(cx + r * cos(a0)), (int)lround(cy - r * sin(a0)),
            (int)lround(cx + r * cos(a1)), (int)lround(cy - r * sin(a1)));
}

// Paint the glyph white-on-black into `dc` at size S. Geometry is authored in
// 16-px units and scaled, so the notification-area icon and the review dump
// are the same drawing rather than two that can drift.
static void paint_glyph(HDC dc, tray_icon_state st, int S) {
    const double s = S / 16.0, cx = 8.0 * s, cy = 8.0 * s;
    RECT full = { 0, 0, S, S };
    FillRect(dc, &full, (HBRUSH)GetStockObject(BLACK_BRUSH));

    int w = (int)lround(1.0 * s); if (w < 1) w = 1;
    HPEN pen = CreatePen(PS_SOLID, w, RGB(255, 255, 255));
    HGDIOBJ oldpen = SelectObject(dc, pen);

    // the core: a 6-unit rounded square, filled unless nothing is loaded
    int l = (int)lround(5.0 * s), t = (int)lround(5.0 * s);
    int r = (int)lround(11.0 * s), b = (int)lround(11.0 * s);
    int rad = (int)lround(4.0 * s);
    HGDIOBJ oldbr = SelectObject(dc, st == TRAY_ICON_IDLE
                                     ? GetStockObject(NULL_BRUSH)
                                     : GetStockObject(WHITE_BRUSH));
    RoundRect(dc, l, t, r, b, rad, rad);
    SelectObject(dc, oldbr);

    // the sweeps, on a null brush so Arc does not close and fill them
    SelectObject(dc, GetStockObject(NULL_BRUSH));
    if (st == TRAY_ICON_RUNNING) {
        for (int i = 0; i < 4; i++)
            arc_seg(dc, cx, cy, 6.0 * s, 45.0 + i * 90.0, 32.0);
    } else {
        arc_seg(dc, cx, cy, 6.0 * s, 45.0, 30.0);
        arc_seg(dc, cx, cy, 6.0 * s, 225.0, 30.0);
    }
    SelectObject(dc, oldpen);
    DeleteObject(pen);
}

static HICON grid_icon(tray_icon_state st) {
    const int S = 16;
    HDC screen = GetDC(NULL);
    HDC dc = CreateCompatibleDC(screen);
    HBITMAP color = CreateCompatibleBitmap(screen, S, S);
    HBITMAP mask = CreateBitmap(S, S, 1, 1, NULL);
    ReleaseDC(NULL, screen);

    HGDIOBJ old = SelectObject(dc, color);
    paint_glyph(dc, st, S);
    SelectObject(dc, old);

    // mask: all zeros = fully opaque square; the black background reads as
    // transparent enough on the taskbar and keeps v1 free of alpha plumbing
    ICONINFO ii = { TRUE, 0, 0, mask, color };
    HICON icon = CreateIconIndirect(&ii);
    DeleteObject(color);
    DeleteObject(mask);
    DeleteDC(dc);
    return icon;
}

// Design-review seam (see tray.h). Writes 32-bit bottom-up BMPs, because they
// need no encoder and every Windows viewer opens them.
bool tray_platform_icon_dump(const char *dir, int px) {
    const char *names[] = { "idle", "loaded", "running" };
    HDC screen = GetDC(NULL);
    bool ok = true;
    for (int i = 0; i < 3 && ok; i++) {
        HDC dc = CreateCompatibleDC(screen);
        HBITMAP bm = CreateCompatibleBitmap(screen, px, px);
        HGDIOBJ old = SelectObject(dc, bm);
        paint_glyph(dc, (tray_icon_state)i, px);
        SelectObject(dc, old);

        BITMAPINFO bi = {0};
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = px;
        bi.bmiHeader.biHeight = px;          // positive = bottom-up
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        DWORD nbytes = (DWORD)px * (DWORD)px * 4;
        void *bits = malloc(nbytes);
        if (!bits || !GetDIBits(dc, bm, 0, (UINT)px, bits, &bi, DIB_RGB_COLORS)) {
            ok = false;
        } else {
            char path[1200];
            snprintf(path, sizeof path, "%s\\tray-%s.bmp", dir, names[i]);
            FILE *f = fopen(path, "wb");
            if (!f) {
                ok = false;
            } else {
                BITMAPFILEHEADER fh = {0};
                fh.bfType = 0x4D42;   // "BM"
                fh.bfOffBits = sizeof fh + sizeof(BITMAPINFOHEADER);
                fh.bfSize = fh.bfOffBits + nbytes;
                fwrite(&fh, sizeof fh, 1, f);
                fwrite(&bi.bmiHeader, sizeof(BITMAPINFOHEADER), 1, f);
                fwrite(bits, 1, nbytes, f);
                fclose(f);
                printf("wrote %s\n", path);
            }
        }
        free(bits);
        DeleteObject(bm);
        DeleteDC(dc);
    }
    ReleaseDC(NULL, screen);
    return ok;
}

static void set_icon(void) {
    HICON ic = grid_icon(tray_icon());
    g_nid.hIcon = ic;
    Shell_NotifyIconA(NIM_MODIFY, &g_nid);
    DestroyIcon(ic);
}

// -------------------------------------------------------------- popup menu

static HMENU build_menu(void) {
    g_nitems = tray_menu_build(g_items, TRAY_MAX_ITEMS);
    HMENU root = CreatePopupMenu();
    HMENU cur = root;
    HMENU stack_parent = NULL;
    int last_pos = -1;

    for (int i = 0; i < g_nitems; i++) {
        tray_item *t = &g_items[i];
        switch (t->kind) {
        case TRAY_K_SEP:
            AppendMenuA(cur, MF_SEPARATOR, 0, NULL);
            break;
        case TRAY_K_SUB_BEGIN: {
            HMENU sub = CreatePopupMenu();
            if (last_pos >= 0)
                ModifyMenuA(cur, (UINT)last_pos, MF_BYPOSITION | MF_POPUP | MF_STRING,
                            (UINT_PTR)sub, g_items[i - 1].label);
            stack_parent = cur;
            cur = sub;
            break;
        }
        case TRAY_K_SUB_END:
            if (stack_parent) { cur = stack_parent; stack_parent = NULL; }
            break;
        default: {
            UINT flags = MF_STRING;
            UINT_PTR id = (UINT_PTR)(MENU_ID_BASE + i);
            if (t->kind == TRAY_K_LABEL) { flags |= MF_GRAYED; id = 0; }
            if (t->kind == TRAY_K_CHECK && t->checked) flags |= MF_CHECKED;
            AppendMenuA(cur, flags, id, t->label);
            last_pos = GetMenuItemCount(cur) - 1;
            break;
        }
        }
    }
    return root;
}

static void pick_model(void) {
    char file[1024] = "";
    OPENFILENAMEA ofn = { .lStructSize = sizeof ofn };
    ofn.hwndOwner = g_hwnd;
    ofn.lpstrFilter = "GGUF models (*.gguf)\0*.gguf\0All files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = sizeof file;
    ofn.lpstrTitle = "Choose a GGUF model for the desktop-managed runner";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameA(&ofn))
        tray_menu_act(TRAY_ACT_PICK_MODEL, 0, file);
    else
        tray_menu_act(TRAY_ACT_PICK_MODEL, 0, NULL);
}

static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM w, LPARAM l) {
    switch (msg) {
    case WM_TRAY_CALLBACK:
        if (LOWORD(l) == WM_RBUTTONUP || LOWORD(l) == WM_LBUTTONUP) {
            HMENU m = build_menu();
            POINT pt;
            GetCursorPos(&pt);
            SetForegroundWindow(h);  // required or the menu won't dismiss
            UINT cmd = (UINT)TrackPopupMenu(m,
                TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON,
                pt.x, pt.y, 0, h, NULL);
            DestroyMenu(m);
            if (cmd >= MENU_ID_BASE) {
                tray_item *t = &g_items[cmd - MENU_ID_BASE];
                if (t->action == TRAY_ACT_PICK_MODEL)
                    pick_model();
                else
                    tray_menu_act(t->action, t->arg, NULL);
            }
            if (tray_should_quit()) PostQuitMessage(0);
            else set_icon();
        }
        return 0;
    case WM_TIMER:
        if (tray_should_quit()) PostQuitMessage(0);
        else set_icon();
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(h, msg, w, l);
}

// -------------------------------------------------------------- autostart
// HKCU\Software\Microsoft\Windows\CurrentVersion\Run, value "XyntetikTray".

#define RUN_KEY "Software\\Microsoft\\Windows\\CurrentVersion\\Run"
#define RUN_VAL "XyntetikTray"
#define RUN_VAL_OLD "GridcoreTray"  // pre-rename value, migrated on sight

// One-time migration of the pre-rename autostart value (Gridcore ->
// Xyntetik): re-register under the new value with the current executable and
// delete the old one, so autostart survives the rename without leaving two
// registrations behind.
static void migrate_old_autostart(void) {
    HKEY k;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_READ | KEY_SET_VALUE,
                      &k) != ERROR_SUCCESS)
        return;
    if (RegQueryValueExA(k, RUN_VAL_OLD, NULL, NULL, NULL, NULL) == ERROR_SUCCESS) {
        char exe[1024], cmd[1100];
        GetModuleFileNameA(NULL, exe, sizeof exe);
        snprintf(cmd, sizeof cmd, "\"%s\" --tray", exe);
        RegSetValueExA(k, RUN_VAL, 0, REG_SZ,
                       (const BYTE *)cmd, (DWORD)strlen(cmd) + 1);
        RegDeleteValueA(k, RUN_VAL_OLD);
    }
    RegCloseKey(k);
}

bool tray_platform_autostart_get(void) {
    migrate_old_autostart();
    HKEY k;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_READ, &k) != ERROR_SUCCESS)
        return false;
    bool present = RegQueryValueExA(k, RUN_VAL, NULL, NULL, NULL, NULL) == ERROR_SUCCESS;
    RegCloseKey(k);
    return present;
}

bool tray_platform_autostart_set(bool on) {
    HKEY k;
    if (RegOpenKeyExA(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_SET_VALUE, &k) != ERROR_SUCCESS)
        return false;
    bool ok;
    if (on) {
        char exe[1024], cmd[1100];
        GetModuleFileNameA(NULL, exe, sizeof exe);
        snprintf(cmd, sizeof cmd, "\"%s\" --tray", exe);
        ok = RegSetValueExA(k, RUN_VAL, 0, REG_SZ,
                            (const BYTE *)cmd, (DWORD)strlen(cmd) + 1) == ERROR_SUCCESS;
    } else {
        LONG rc = RegDeleteValueA(k, RUN_VAL);
        ok = rc == ERROR_SUCCESS || rc == ERROR_FILE_NOT_FOUND;
    }
    RegCloseKey(k);
    return ok;
}

// -------------------------------------------------------------- main loop

int tray_platform_run(void) {
    migrate_old_autostart();
    FreeConsole();  // detach from any console we were launched from

    WNDCLASSA wc = { .lpfnWndProc = wndproc,
                     .hInstance = GetModuleHandleA(NULL),
                     .lpszClassName = "XyntetikTrayWnd" };
    RegisterClassA(&wc);
    g_hwnd = CreateWindowA(wc.lpszClassName, "xyntetik-tray", 0, 0, 0, 0, 0,
                           HWND_MESSAGE, NULL, wc.hInstance, NULL);
    if (!g_hwnd) return 1;

    memset(&g_nid, 0, sizeof g_nid);
    g_nid.cbSize = sizeof g_nid;
    g_nid.hWnd = g_hwnd;
    g_nid.uID = TRAY_ICON_ID;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAY_CALLBACK;
    g_nid.hIcon = grid_icon(tray_icon());
    snprintf(g_nid.szTip, sizeof g_nid.szTip, "xyntetik-runner");
    Shell_NotifyIconA(NIM_ADD, &g_nid);

    SetTimer(g_hwnd, 1, 5000, NULL);  // badge refresh while menu is closed

    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    Shell_NotifyIconA(NIM_DELETE, &g_nid);
    DestroyWindow(g_hwnd);
    return 0;
}

#endif // _WIN32
