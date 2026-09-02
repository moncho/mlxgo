//go:build mlx

package mlx

/*
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <mlx/c/error.h>

static pthread_mutex_t mlxgo_error_mu = PTHREAD_MUTEX_INITIALIZER;
static char* mlxgo_last_error = NULL;

static void mlxgo_error_handler(const char* msg, void* data) {
	(void)data;

	pthread_mutex_lock(&mlxgo_error_mu);
	free(mlxgo_last_error);
	mlxgo_last_error = NULL;
	if (msg != NULL) {
		mlxgo_last_error = strdup(msg);
	}
	pthread_mutex_unlock(&mlxgo_error_mu);
}

static void mlxgo_install_error_handler(void) {
	mlx_set_error_handler(mlxgo_error_handler, NULL, NULL);
}

static void mlxgo_clear_last_error(void) {
	pthread_mutex_lock(&mlxgo_error_mu);
	free(mlxgo_last_error);
	mlxgo_last_error = NULL;
	pthread_mutex_unlock(&mlxgo_error_mu);
}

static char* mlxgo_take_last_error(void) {
	pthread_mutex_lock(&mlxgo_error_mu);
	char* msg = mlxgo_last_error;
	mlxgo_last_error = NULL;
	pthread_mutex_unlock(&mlxgo_error_mu);
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
