FIGURES = overview.png sensitivity_analysis.png \
synthetic_inversion_uks.png time_frequency_analysis.png cumulative_heat.png

%.png : %.py
	python $<

all: $(FIGURES)
	@echo $(FIGURES)

clean:
	@rm data/uks.nc
	@rm $(FIGURES)
