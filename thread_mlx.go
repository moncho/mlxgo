//go:build mlx

package mlx

/*
#include <pthread.h>

static pthread_key_t mlxgo_worker_key;
static pthread_once_t mlxgo_worker_key_once = PTHREAD_ONCE_INIT;
static int mlxgo_worker_marker;

static void mlxgo_make_worker_key(void) {
	(void)pthread_key_create(&mlxgo_worker_key, NULL);
}

static void mlxgo_mark_worker_thread(void) {
	pthread_once(&mlxgo_worker_key_once, mlxgo_make_worker_key);
	(void)pthread_setspecific(mlxgo_worker_key, &mlxgo_worker_marker);
}

static int mlxgo_is_worker_thread(void) {
	pthread_once(&mlxgo_worker_key_once, mlxgo_make_worker_key);
	return pthread_getspecific(mlxgo_worker_key) != NULL;
}
*/
import "C"

import (
	"runtime"
	"sync"
)

type mlxThreadCall struct {
	run  func()
	done chan struct{}
}

var mlxThread = struct {
	once sync.Once
	jobs chan mlxThreadCall
}{}

func startMLXThread() {
	mlxThread.once.Do(func() {
		mlxThread.jobs = make(chan mlxThreadCall)
		go func() {
			runtime.LockOSThread()
			C.mlxgo_mark_worker_thread()
			for call := range mlxThread.jobs {
				call.run()
				close(call.done)
			}
		}()
	})
}

func onMLXThread() bool {
	return C.mlxgo_is_worker_thread() != 0
}

func runMLX(fn func() error) (err error) {
	if onMLXThread() {
		return fn()
	}

	startMLXThread()

	var panicValue any
	done := make(chan struct{})
	mlxThread.jobs <- mlxThreadCall{
		run: func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					panicValue = recovered
				}
			}()
			err = fn()
		},
		done: done,
	}
	<-done

	if panicValue != nil {
		panic(panicValue)
	}
	return err
}

func runMLXValue[T any](fn func() (T, error)) (T, error) {
	var value T
	err := runMLX(func() error {
		var err error
		value, err = fn()
		return err
	})
	return value, err
}
