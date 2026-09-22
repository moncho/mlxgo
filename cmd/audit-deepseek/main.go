// audit-deepseek inspects pinned checkpoint metadata, never tensor payloads.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/moncho/mlxgo/internal/deepseekaudit"
)

func main() {
	remote := flag.Bool("remote", false, "fetch pinned metadata using bounded HTTP range requests")
	snapshot := flag.String("snapshot", "", "read an existing .json.gz metadata snapshot offline")
	save := flag.String("save-snapshot", "", "save downloaded metadata to a new .json.gz file")
	out := flag.String("out", "", "write JSON report to a new file (default stdout)")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, *remote, *snapshot, *save, *out, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(ctx context.Context, remote bool, path, save, out string, stdout, stderr io.Writer) error {
	if remote == (path != "") || (save != "" && !remote) {
		return fmt.Errorf("choose exactly one of -remote or -snapshot; -save-snapshot requires -remote")
	}
	if save != "" && save == out {
		return fmt.Errorf("snapshot and report must use different paths")
	}
	var s deepseekaudit.Snapshot
	var err error
	if remote {
		s, err = deepseekaudit.Download(ctx, func(file string) { fmt.Fprintln(stderr, "Inspecting header:", file) })
	} else {
		f, e := os.Open(path)
		if e != nil {
			return e
		}
		s, err = deepseekaudit.ReadSnapshot(f)
		f.Close()
	}
	if err != nil {
		return err
	}
	r, err := deepseekaudit.Audit(s)
	if err != nil {
		return err
	}
	if save != "" {
		if err = writeNew(save, func(w io.Writer) error { return deepseekaudit.WriteSnapshot(w, s) }); err != nil {
			return err
		}
	}
	emit := func(w io.Writer) error { e := json.NewEncoder(w); e.SetIndent("", "  "); return e.Encode(r) }
	if out != "" {
		err = writeNew(out, emit)
	} else {
		err = emit(stdout)
	}
	if err != nil {
		return err
	}
	if !r.MetadataValid {
		return fmt.Errorf("audit found %d tensor mapping issues; see report", len(r.Issues))
	}
	return nil
}

func writeNew(path string, write func(io.Writer) error) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	err = errors.Join(write(f), f.Close())
	if err != nil {
		_ = os.Remove(path)
	}
	return err
}
