// model_file_identity() must be able to key the files the runner actually
// serves. The 2026-08-04 shared-weights defect was exactly this failing at
// scale: MinGW's stat() returns EOVERFLOW for files past 2 GB, the identity
// was lost, every --parallel slot loaded privately and re-decided its own
// CPU/GPU split under the previous slot's VRAM pressure. test.gguf (370 KB)
// can never catch that, so this test manufactures a file at real-checkpoint
// size — 5 GB, sparse where the OS allows it — and demands a usable identity.
// It skips (exit 0, loudly) only when the disk cannot spare the space.
#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#endif

static int g_fail = 0;

static void ck(int cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail = 1; }
}

// 5 GB: comfortably past both the 2 GB signed-32-bit st_size cliff and the
// 4 GB unsigned one, so a truncated size cannot alias back to the real value.
static const uint64_t BIG = 5ull * 1024 * 1024 * 1024;

// Extend path to BIG bytes. Returns false when the filesystem refuses —
// which the caller treats as "disk can't spare it", not as a failure.
static bool make_big(const char *path) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, NULL);
    if (h == INVALID_HANDLE_VALUE) return false;
    // Sparse first so NTFS doesn't reserve 5 GB of real clusters; if the
    // volume can't do sparse files SetEndOfFile still works, it just needs
    // the space (and failing that is the skip path).
    DWORD ret = 0;
    DeviceIoControl(h, FSCTL_SET_SPARSE, NULL, 0, NULL, 0, &ret, NULL);
    LARGE_INTEGER sz;
    sz.QuadPart = (LONGLONG)BIG;
    bool ok = SetFilePointerEx(h, sz, NULL, FILE_BEGIN) && SetEndOfFile(h);
    CloseHandle(h);
    return ok;
#else
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0) return false;
    bool ok = ftruncate(fd, (off_t)BIG) == 0;
    close(fd);
    return ok;
#endif
}

int main(void) {
    const char *path = "test-file-identity.tmp";

    if (!make_big(path)) {
        remove(path);
        fprintf(stderr, "skip: test-file-identity (could not create a 5 GB "
                        "file here — disk full or filesystem limit)\n");
        return 0;
    }

    uint64_t size = 0, ino = 0;
    int64_t mtime = 0, ctime_ = 0;
    bool ok = model_file_identity(path, "host weights", &size, &ino,
                                  &mtime, &ctime_);
    ck(ok, "a 5 GB file can be keyed by file identity");
    if (ok) {
        ck(size == BIG, "identity carries the real 64-bit size");
        ck(ino != 0, "identity carries a real file index");
        ck(mtime > 0, "identity carries a real mtime");
    }

    // Same file asked twice must key identically — this equality is what
    // mw_find/shared_matches use to decide two loads may share weights.
    if (ok) {
        uint64_t size2 = 0, ino2 = 0;
        int64_t mtime2 = 0, ctime2 = 0;
        bool ok2 = model_file_identity(path, "host weights", &size2, &ino2,
                                       &mtime2, &ctime2);
        ck(ok2, "second lookup of the same file succeeds");
        ck(ok2 && size2 == size && ino2 == ino &&
           mtime2 == mtime && ctime2 == ctime_,
           "two lookups of one unchanged file agree on every field");
    }

    remove(path);
    if (!g_fail) printf("file identity ok at real-checkpoint size\n");
    return g_fail;
}
