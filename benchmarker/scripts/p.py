import os
import pickle

# Spécifie le répertoire contenant les fichiers
directory = 'data/instances'  # À modifier selon ton cas


# Fonction pour déterminer la précision
def get_precision(system):
    return 3 if system in [3, 8] else 2

# Traitement des fichiers
for system in [i for i in range(0,9)]:
    file_path = os.path.join(directory, f"{str(system)}.pckl")
    if os.path.isfile(file_path):
        
        try:
            with open(file_path, 'rb') as f:
                instance_dict = pickle.load(f)

            instance_dict['precision'] = get_precision(system)
            print(instance_dict)

            with open(file_path, 'wb') as f:
                pickle.dump(instance_dict, f)

        except Exception as e:
            print(f"Błąd przy pliku {file_path}: {e}")
