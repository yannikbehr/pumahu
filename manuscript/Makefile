pdf:
	pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript

figs:
	$(MAKE) -C figures all

pandoc:
	pandoc -f latex+latex_macros --bibliography=zotero.bib -t odt -o manuscript.odt manuscript.tex

clean:
	@rm *.bbl *.aux *.blg *.log
