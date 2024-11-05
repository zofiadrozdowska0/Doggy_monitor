import os

# sciezka do folderu z plikami .txt
folder_path = 'datasets/data/labels/test'

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as file:
            data = file.read()

        # > 1.0 na 0.99, <0.0 na 0.0
        modified_data = []
        for value in data.split():
            try:
                if value != '2':
                    float_value = float(value)
                    if float_value > 1.0:
                        modified_data.append('0.99')
                    elif float_value < 0.0:
                        modified_data.append('0.0')
                    else:
                        modified_data.append(value)
                else:
                    modified_data.append(value)
            except ValueError:
                modified_data.append(value)

        # Zapis
        with open(file_path, 'w') as file:
            file.write(' '.join(modified_data))

print("Wszystkie pliki zostały przetworzone.")
