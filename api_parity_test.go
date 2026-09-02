package mlx

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestStubAndNativeAPIsStayInSync(t *testing.T) {
	native, err := exportedAPI("array_mlx.go", "closure_mlx.go")
	if err != nil {
		t.Fatal(err)
	}
	stub, err := exportedAPI("array_stub.go", "closure_stub.go")
	if err != nil {
		t.Fatal(err)
	}

	if reflect.DeepEqual(native, stub) {
		return
	}

	t.Fatalf("stub/native API mismatch\nnative only:\n%s\nstub only:\n%s\nsignature differences:\n%s",
		formatAPIOnly(native, stub),
		formatAPIOnly(stub, native),
		formatAPIDifferences(native, stub),
	)
}

func exportedAPI(files ...string) (map[string]string, error) {
	fset := token.NewFileSet()
	api := make(map[string]string)
	for _, file := range files {
		parsed, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			return nil, err
		}
		for _, decl := range parsed.Decls {
			switch decl := decl.(type) {
			case *ast.GenDecl:
				if decl.Tok != token.TYPE {
					continue
				}
				for _, spec := range decl.Specs {
					typeSpec := spec.(*ast.TypeSpec)
					if typeSpec.Name.IsExported() {
						api["type "+typeSpec.Name.Name] = "type"
					}
				}
			case *ast.FuncDecl:
				if !decl.Name.IsExported() {
					continue
				}
				if decl.Recv == nil {
					api["func "+decl.Name.Name] = funcSignature(decl.Type)
					continue
				}
				recv := receiverName(decl.Recv)
				if ast.IsExported(recv) {
					api["method "+recv+"."+decl.Name.Name] = funcSignature(decl.Type)
				}
			}
		}
	}
	return api, nil
}

func funcSignature(fn *ast.FuncType) string {
	params := fieldListSignature(fn.Params)
	results := fieldListSignature(fn.Results)
	if len(results) == 0 {
		return fmt.Sprintf("func(%s)", strings.Join(params, ", "))
	}
	if len(results) == 1 {
		return fmt.Sprintf("func(%s) %s", strings.Join(params, ", "), results[0])
	}
	return fmt.Sprintf("func(%s) (%s)", strings.Join(params, ", "), strings.Join(results, ", "))
}

func fieldListSignature(fields *ast.FieldList) []string {
	if fields == nil {
		return nil
	}
	var out []string
	for _, field := range fields.List {
		count := len(field.Names)
		if count == 0 {
			count = 1
		}
		typ := exprString(field.Type)
		for i := 0; i < count; i++ {
			out = append(out, typ)
		}
	}
	return out
}

func receiverName(fields *ast.FieldList) string {
	if fields == nil || len(fields.List) == 0 {
		return ""
	}
	typ := fields.List[0].Type
	if star, ok := typ.(*ast.StarExpr); ok {
		typ = star.X
	}
	switch typ := typ.(type) {
	case *ast.Ident:
		return typ.Name
	default:
		return exprString(typ)
	}
}

func exprString(expr ast.Expr) string {
	var buf bytes.Buffer
	_ = printer.Fprint(&buf, token.NewFileSet(), expr)
	return buf.String()
}

func formatAPIOnly(left, right map[string]string) string {
	var keys []string
	for key := range left {
		if _, ok := right[key]; !ok {
			keys = append(keys, key+" "+left[key])
		}
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		return "  none"
	}
	return "  " + strings.Join(keys, "\n  ")
}

func formatAPIDifferences(left, right map[string]string) string {
	var diffs []string
	for key, leftSig := range left {
		rightSig, ok := right[key]
		if ok && leftSig != rightSig {
			diffs = append(diffs, fmt.Sprintf("%s: native %s / stub %s", key, leftSig, rightSig))
		}
	}
	sort.Strings(diffs)
	if len(diffs) == 0 {
		return "  none"
	}
	return "  " + strings.Join(diffs, "\n  ")
}
