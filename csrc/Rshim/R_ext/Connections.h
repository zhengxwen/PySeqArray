// Rshim/R_ext/Connections.h — stub for R's connection API.  PySeqArray replaces
// R connections with stdio/zlib in B3 (VCF I/O); for now the types/decls exist so
// the conversion TUs compile, and the read path (which does not use them) links.
#ifndef PYSEQARRAY_RSHIM_REXT_CONNECTIONS_H_
#define PYSEQARRAY_RSHIM_REXT_CONNECTIONS_H_

#include "../Rinternals.h"
#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#define R_CONNECTIONS_VERSION 1

typedef struct Rconn *Rconnection;

// Connection struct exposing the members SeqArray's VCF I/O touches, including
// R's internal pushback-buffer fields (buff/buff_pos/buff_stored_len) and
// EOF_signalled.  PySeqArray backs it with a zlib gzFile (handles plain + gzip).
struct Rconn {
    char  *class_name;
    char  *description;
    int    text;
    int    isopen;
    int    EOF_signalled;
    char  *buff;             // pushback buffer (unused here -> NULL)
    size_t buff_pos;
    size_t buff_stored_len;
    size_t (*read)(void *, size_t, size_t, Rconnection);
    size_t (*write)(const void *, size_t, size_t, Rconnection);
    int    (*fflush)(Rconnection);
    int    (*vfprintf)(Rconnection, const char *, va_list);
    int    (*fgetc)(Rconnection);
    void  *_private;
};

Rconnection R_GetConnection(SEXP sConn);
size_t R_ReadConnection(Rconnection con, void *buf, size_t n);
size_t R_WriteConnection(Rconnection con, const void *buf, size_t n);

// PySeqArray helpers (csrc/conn.cpp)
#ifdef __cplusplus
extern "C" {
#endif
// open a gzFile-backed read/write connection for a path; mode "rb" or "wb".
Rconnection Rsh_open_connection(const char *path, const char *mode);
void Rsh_close_connections(void);  // close + free all open shim connections
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
}
#endif

#endif
