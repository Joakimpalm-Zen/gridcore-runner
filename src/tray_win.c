// Windows tray backend: Shell_NotifyIcon + a hidden message-only window.
// The runner stays a console-subsystem binary (no -mwindows — that would
// break every CLI mode); when launched with --tray from Explorer the
// console is released with FreeConsole so no black window lingers.
#ifdef _WIN32
#include "tray.h"

#include <windows.h>
#include <shellapi.h>
#include <commdlg.h>
#include <stdio.h>
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
// Same code-drawn 3×3 grid as macOS, painted into a 16×16 HICON.
static HICON grid_icon(bool filled) {
    const int S = 16;
    HDC screen = GetDC(NULL);
    HDC dc = CreateCompatibleDC(screen);
    HBITMAP color = CreateCompatibleBitmap(screen, S, S);
    HBITMAP mask = CreateBitmap(S, S, 1, 1, NULL);
    ReleaseDC(NULL, screen);

    HGDIOBJ old = SelectObject(dc, color);
    RECT full = { 0, 0, S, S };
    FillRect(dc, &full, (HBRUSH)GetStockObject(BLACK_BRUSH));
    HBRUSH white = (HBRUSH)GetStockObject(WHITE_BRUSH);
    HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
    HGDIOBJ oldpen = SelectObject(dc, pen);
    SelectObject(dc, GetStockObject(NULL_BRUSH));
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            int x = 1 + c * 5, y = 1 + r * 5;
            RECT cr = { x, y, x + 4, y + 4 };
            if (r == 1 && c == 1 && filled)
                FillRect(dc, &cr, white);
            else
                Rectangle(dc, cr.left, cr.top, cr.right, cr.bottom);
        }
    SelectObject(dc, oldpen);
    DeleteObject(pen);
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

static void set_icon(void) {
    HICON ic = grid_icon(tray_any_running());
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
// HKCU\Software\Microsoft\Windows\CurrentVersion\Run, value "GridcoreTray".

#define RUN_KEY "Software\\Microsoft\\Windows\\CurrentVersion\\Run"
#define RUN_VAL "GridcoreTray"

bool tray_platform_autostart_get(void) {
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
    FreeConsole();  // detach from any console we were launched from

    WNDCLASSA wc = { .lpfnWndProc = wndproc,
                     .hInstance = GetModuleHandleA(NULL),
                     .lpszClassName = "GridcoreTrayWnd" };
    RegisterClassA(&wc);
    g_hwnd = CreateWindowA(wc.lpszClassName, "gridcore-tray", 0, 0, 0, 0, 0,
                           HWND_MESSAGE, NULL, wc.hInstance, NULL);
    if (!g_hwnd) return 1;

    memset(&g_nid, 0, sizeof g_nid);
    g_nid.cbSize = sizeof g_nid;
    g_nid.hWnd = g_hwnd;
    g_nid.uID = TRAY_ICON_ID;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAY_CALLBACK;
    g_nid.hIcon = grid_icon(tray_any_running());
    snprintf(g_nid.szTip, sizeof g_nid.szTip, "gridcore-runner");
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
