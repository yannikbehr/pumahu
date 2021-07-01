pdf:
	pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript

figs:
	$(MAKE) -C figures all

clean:
	@rm *.bbl *.aux *.blg *.log
