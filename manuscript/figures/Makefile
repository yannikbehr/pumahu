
FIGURES = sensitivity_analysis.png \
		  synthetic_inversion_uks.png

%.png : %.py
	python $<

all: $(FIGURES)
	@echo $(FIGURES)

clear:
	@rm $(FIGURES)
