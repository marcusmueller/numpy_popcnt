// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019, Marcus Müller / CEL
#include "Python.h"
#include <popcntintrin.h>
#include "numpy/ndarraytypes.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/ufuncobject.h"
/*
 * numpy_popcnt.c
 * Popcnt for uint32 arrays based on SSE4A (effectively: SSE4.2) popcnt instr
 */

static PyMethodDef PopcntMethods[] = {{NULL, NULL, 0, NULL}};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void uint32_popcnt(char **args, npy_intp *dimensions, npy_intp *steps,
                          void *data) {
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  uint32_t tmp;

  for (i = 0; i < n; i++) {
    tmp = *(uint32_t *)in;
    *((uint8_t *)out) = (uint8_t)(_mm_popcnt_u32(tmp));
    in += in_step;
    out += out_step;
  }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&uint32_popcnt};

/* These are the input and return dtypes of popcnt.*/
static char types[2] = {NPY_UINT32, NPY_UINT8};

static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "popcnt",
                                       NULL,
                                       -1,
                                       PopcntMethods,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL};

PyMODINIT_FUNC PyInit_popcnt(void) {
  PyObject *m, *popcnt, *d;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }

  import_array();
  import_umath();

  popcnt = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1, PyUFunc_None,
                                   "popcnt", "take the population count of an uint32 input array and return it as uint8 array", 0);

  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "popcnt", popcnt);
  Py_DECREF(popcnt);

  return m;
}
