//go:build !darwin && !linux

package main

func peakRSS() uint64 { return 0 }
