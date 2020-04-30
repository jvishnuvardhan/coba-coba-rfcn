import re
model_path = "D: \Workspace\College\Semester 8\Tugas Akhir\Keras-RFCN-master\Keras-RFCN-master\ModelData\logs\bdd20200422T1046\Keras-RFCN_bdd_0005.h5
regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/Keras-RFCN\_\w+(\d{4})\.h5"
m = re.match(regex, model_path)
print("ayo napa mulai dari epoch 0", m)
if m:
    self.epoch = int(m.group(6)) + 1
    print(self.epoch)
