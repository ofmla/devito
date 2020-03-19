#!/bin/bash
gmtset PS_MEDIA = A3
gmt gmtset MAP_LABEL_OFFSET -0.5i
gmt gmtset MAP_LABEL_OFFSET 0.3c
gmt gmtset FONT_LABEL 7p
gmt gmtset FONT_ANNOT_PRIMARY=6
gmt gmtset PS_PAGE_ORIENTATION=LANDSCAPE
gmtset MAP_FRAME_PEN=thin,black
gmt gmtset MAP_GRID_PEN_PRIMARY=thinner,black,--
gmtset MAP_TICK_LENGTH_PRIMARY=-0.05
gmtset MAP_TICK_LENGTH_SECONDARY=-0.025
region="-R1/1273/1/20002"
projection="-JX1.8i/-1i"
open="-K -V"
add="-O -K -V"
close="-O -V"
psFile="shot.ps"
pdfFile="shot.pdf"
epsFile="shot.eps"
contourPlot=0
dx=1.0
dz=1.92
velFile1='2D_shot_variable_density_all_receivers-4hz-multiples.file'

gmt xyz2grd $velFile1 -ZLBf $region -I${dx}/${dz} -V -GvelGrid.grd 
gmt grdinfo -L1 -L2 -M velGrid.grd
gmt makecpt -T-3.866/3.636/0.88 -D -Z -I -Cgray > seismicVels.cpt
gmt grdimage velGrid.grd $region $projection -CseismicVels.cpt -Y4i $open > $psFile
if [ $contourPlot -eq 1 ]
then
gmt pscontour tmpVel.beam -A500 -W2.5p $region $projection $add >> $psFile
fi
gmt psbasemap $region $projection -B200:"Traces":/5000:"Time (sec)":WN $add >> $psFile
gmt psbasemap $region $projection -B0SE $add >> $psFile
gmt psscale -CseismicVels.cpt -D0.9i/-0.05i/1.8i/0.06ih --MAP_LABEL_OFFSET=1p -Ba1.76::/::wSen  $close  >> $psFile

ps2eps -R=+ -l -B -r 600 $psFile > $epsFile
epstopdf $epsFile > $pdfFile
#rm $psFile $epsFile
evince $pdfFile &

