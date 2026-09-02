//go:build mlx

package mlx

/*
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <mlx/c/error.h>

static pthread_key_t mlxgo_error_key;
static pthread_once_t mlxgo_error_key_once = PTHREAD_ONCE_INIT;

static void mlxgo_free_thread_error(void* ptr) {
	free(ptr);
}

static void mlxgo_make_error_key(void) {
	(void)pthread_key_create(&mlxgo_error_key, mlxgo_free_thread_error);
}

static void mlxgo_set_thread_error(const char* msg) {
	pthread_once(&mlxgo_error_key_once, mlxgo_make_error_key);

	char* old = (char*)pthread_getspecific(mlxgo_error_key);
	free(old);

	char* next = NULL;
	if (msg != NULL) {
		next = strdup(msg);
	}
	(void)pthread_setspecific(mlxgo_error_key, next);
}

static void mlxgo_error_handler(const char* msg, void* data) {
	(void)data;
	mlxgo_set_thread_error(msg);
}

static void mlxgo_install_error_handler(void) {
	mlx_set_error_handler(mlxgo_error_handler, NULL, NULL);
}

static void mlxgo_clear_last_error(void) {
	mlxgo_set_thread_error(NULL);
}

static char* mlxgo_take_last_error(void) {
	pthread_once(&mlxgo_error_key_once, mlxgo_make_error_key);

	char* msg = (char*)pthread_getspecific(mlxgo_error_key);
	(void)pthread_setspecific(mlxgo_error_key, NULL);
	return msg;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func init() {
	C.mlxgo_install_error_handler()
}

func clearMLXError() {
	C.mlxgo_clear_last_error()
}

func mlxError(name string, code int) error {
	text := takeMLXError()
	if text == "" {
		return fmt.Errorf("mlxgo: %s failed with code %d", name, code)
	}
	return fmt.Errorf("mlxgo: %s failed with code %d: %s", name, code, text)
}

func mlxEmptyHandleError(name, kind string) error {
	text := takeMLXError()
	if text == "" {
		return fmt.Errorf("mlxgo: %s returned an empty %s", name, kind)
	}
	return fmt.Errorf("mlxgo: %s returned an empty %s: %s", name, kind, text)
}

func takeMLXError() string {
	msg := C.mlxgo_take_last_error()
	if msg == nil {
		return ""
	}
	defer C.free(unsafe.Pointer(msg))
	return C.GoString(msg)
}
