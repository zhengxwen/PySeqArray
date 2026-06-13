// conn.cpp — real connection layer for PySeqArray, backing R's Rconnection with a
// zlib gzFile (transparently handles plain and gzip-compressed VCF).  Replaces
// the throwing stubs; used by SeqArray's VCF import/export (ConvVCF2GDS /
// ConvGDS2VCF).
//
// The "connection" SEXP passed in by the Python orchestration is a STRSXP whose
// first element is the file path.  SEQ_VCF_Parse skips the VCF header itself, so
// the file is simply opened at position 0.

#include "Rshim/Rshim_cpp.h"
#include "Rshim/R_ext/Connections.h"

#include <zlib.h>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <vector>
#include <string>

struct ConnData { gzFile gz; bool writing; };

static std::vector<Rconnection> g_conns;

static size_t conn_read(void *ptr, size_t sz, size_t n, Rconnection con)
{
    ConnData *d = (ConnData*)con->_private;
    int r = gzread(d->gz, ptr, (unsigned)(sz * n));
    return r > 0 ? (size_t)r : 0;
}
static size_t conn_write(const void *ptr, size_t sz, size_t n, Rconnection con)
{
    ConnData *d = (ConnData*)con->_private;
    int w = gzwrite(d->gz, ptr, (unsigned)(sz * n));
    return w > 0 ? (size_t)w : 0;
}
static int conn_fflush(Rconnection con)
{
    ConnData *d = (ConnData*)con->_private;
    return gzflush(d->gz, Z_SYNC_FLUSH);
}
static int conn_vfprintf(Rconnection con, const char *fmt, va_list ap)
{
    char buf[4096];
    int m = vsnprintf(buf, sizeof buf, fmt, ap);
    if (m < 0) return m;
    ConnData *d = (ConnData*)con->_private;
    if ((size_t)m < sizeof buf) {
        gzwrite(d->gz, buf, (unsigned)m);
    } else {
        std::vector<char> big(m + 1);
        va_list ap2; va_copy(ap2, ap);
        vsnprintf(&big[0], big.size(), fmt, ap2);
        va_end(ap2);
        gzwrite(d->gz, &big[0], (unsigned)m);
    }
    return m;
}

extern "C" Rconnection Rsh_open_connection(const char *path, const char *mode)
{
    gzFile gz = gzopen(path, mode);
    if (!gz) throw Rsh_error(std::string("cannot open file: ") + path);
    Rconnection con = new Rconn();
    memset(con, 0, sizeof(*con));
    ConnData *d = new ConnData();
    d->gz = gz;
    d->writing = (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL);
    con->_private = d;
    con->text = 1;
    con->isopen = 1;
    con->EOF_signalled = 0;
    con->buff = NULL; con->buff_pos = 0; con->buff_stored_len = 0;
    con->read = conn_read;
    con->write = conn_write;
    con->fflush = conn_fflush;
    con->vfprintf = conn_vfprintf;
    con->fgetc = NULL;
    g_conns.push_back(con);
    return con;
}

// The connection SEXP is a STRSXP{ path } (read) or { path, mode } (e.g. "wb").
extern "C" Rconnection R_GetConnection(SEXP sConn)
{
    if (sConn == NULL || TYPEOF(sConn) != STRSXP || Rf_xlength(sConn) < 1)
        throw Rsh_error("R_GetConnection: expected a file-path string");
    const char *mode = (Rf_xlength(sConn) >= 2) ? R_CHAR(STRING_ELT(sConn, 1)) : "rb";
    return Rsh_open_connection(R_CHAR(STRING_ELT(sConn, 0)), mode);
}

extern "C" size_t R_ReadConnection(Rconnection con, void *buf, size_t n)
{
    return conn_read(buf, 1, n, con);
}
extern "C" size_t R_WriteConnection(Rconnection con, const void *buf, size_t n)
{
    return conn_write(buf, 1, n, con);
}

extern "C" void Rsh_close_connections(void)
{
    for (size_t i = 0; i < g_conns.size(); i++) {
        Rconnection con = g_conns[i];
        ConnData *d = (ConnData*)con->_private;
        if (d) { if (d->gz) gzclose(d->gz); delete d; }
        delete con;
    }
    g_conns.clear();
}
