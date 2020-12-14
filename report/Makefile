
.PHONY: default clean report.pdf

default: report.pdf

report.pdf: literature.bib
	latexmk -pdf -shell-escape report.tex

clean:
	rm -f *.pdf *.out *aux *bbl *blg *log *toc *.ptb *.tod *.fls *.fdb_latexmk *.lof *.xcp *.synctex.gz *.pyg