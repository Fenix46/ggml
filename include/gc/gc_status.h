#pragma once

// ── gc_status_t — common error return type for all public C API functions ────
//
// All public API functions return gc_status_t.  Values ≥ 0 are success;
// negative values are errors.  GC_OK is the only success code.
//
// This header has NO dependencies — it can be included by any subsystem header.

typedef enum gc_status_t {
    GC_OK              =  0,  // success
    GC_ERR_GENERIC     = -1,  // unspecified error
    GC_ERR_IO          = -2,  // I/O error (file read/write)
    GC_ERR_ALLOC       = -3,  // memory allocation failure
    GC_ERR_INVALID     = -4,  // invalid argument / state
    GC_ERR_UNSUPPORTED = -5,  // feature not supported
    GC_ERR_NOT_FOUND   = -6,  // resource not found
    GC_ERR_BUSY        = -7,  // resource in use
} gc_status_t;

// Helper: return true if status indicates success.
static inline int gc_ok(gc_status_t s) { return s == GC_OK; }

// Return a human-readable string for a gc_status_t value.
static inline const char * gc_status_to_string(gc_status_t s) {
    switch (s) {
        case GC_OK:              return "success";
        case GC_ERR_GENERIC:     return "generic error";
        case GC_ERR_IO:          return "I/O error";
        case GC_ERR_ALLOC:       return "allocation failure";
        case GC_ERR_INVALID:     return "invalid argument";
        case GC_ERR_UNSUPPORTED: return "unsupported feature";
        case GC_ERR_NOT_FOUND:   return "not found";
        case GC_ERR_BUSY:        return "resource busy";
        default:                 return "unknown status";
    }
}
