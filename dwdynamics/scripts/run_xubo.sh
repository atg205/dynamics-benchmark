#!/bin/bash
# Sprawdzenie liczby argumentów
if [ "$#" -ne 3 ]; then echo "Użycie: $0 <input_id> <prec> 
  <timepoints>" exit 1
fi 

input_id="$1" prec="$2" timepoints="$3"
# Tworzenie katalogu wyjściowego, jeśli nie istnieje
mkdir -p "/home/atg205/Documents/__Dokumente/Uni/UPMC/stage gl/DWaveDynamics2/data/xubo/output/${input_id}"
# Uruchomienie komendy
/home/atg205/Documents/__Dokumente/Uni/UPMC/stage\ gl/xubo-master/xubo_ising --all-states "/home/atg205/Documents/__Dokumente/Uni/UPMC/stage gl/DWaveDynamics2/data/xubo/ising/${input_id}/precision_${prec}_timepoints_${timepoints}.ising" > "/home/atg205/Documents/__Dokumente/Uni/UPMC/stage gl/DWaveDynamics2/data/xubo/output/${input_id}/precision_${prec}_timepoints_${timepoints}.xubo"
