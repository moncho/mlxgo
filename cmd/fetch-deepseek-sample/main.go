// fetch-deepseek-sample downloads bounded real tensor samples, not a model.
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
	out := flag.String("out", "", "new output directory (parent must exist)")
	set := flag.String("set", "projections", "projections (42.5 MB), expert (18.8 MB), attention (126.8 MB), or compressed-attention (142.9 MB)")
	reuse := flag.String("reuse", "", "existing sample directory to reuse verified files from (expert or attention)")
	flag.Parse()
	if *out == "" || flag.NArg() != 0 || (*set != "projections" && *set != "expert" && *set != "attention" && *set != "compressed-attention") || ((*set == "projections" || *set == "compressed-attention") && *reuse != "") {
		flag.Usage()
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	progress := func(s string) { fmt.Fprintln(os.Stderr, s) }
	var err error
	if *set == "compressed-attention" {
		err = deepseekaudit.DownloadCompressedAttentionSample(ctx, *out, progress)
	} else if *set == "attention" {
		err = deepseekaudit.DownloadAttentionSample(ctx, *out, *reuse, progress)
	} else if *set == "expert" {
		err = deepseekaudit.DownloadExpertSample(ctx, *out, *reuse, progress)
	} else {
		err = deepseekaudit.DownloadSample(ctx, *out, progress)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("Saved", *set, "tensors and manifest.json to", *out)
	fmt.Println("Raw validation samples only; not a loadable checkpoint.")
}
