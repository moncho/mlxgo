// fetch-deepseek-sample downloads three real weight/scale pairs, not a model.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"

	"github.com/moncho/mlxgo/internal/deepseekaudit"
)

func main() {
	out := flag.String("out", "", "new output directory (parent must exist); downloads 42,478,080 payload bytes")
	flag.Parse()
	if *out == "" || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := deepseekaudit.DownloadSample(ctx, *out, func(s string) { fmt.Fprintln(os.Stderr, s) }); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("Saved three weight/scale pairs and manifest.json to", *out)
	fmt.Println("Raw validation samples only; not a loadable checkpoint.")
}
