ant = hornRidge('FlareWidth',270e-3,'FlareHeight',146.2e-3,'FlareLength',225e-3,'NumFlares',2,'RidgeGap',2e-3,'Length',15e-3,'Height',40e-3,'Width',40e-3);
show(ant);
p = PatternPlotOptions(MagnitudeScale=[0 20]);
figure
pattern(ant,10e9,PatternOptions=p);