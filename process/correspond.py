import csv
import os
import xlrd2


data = {}
corre_csv = xlrd2.open_workbook("interpretation.xlsx")
sheet = corre_csv.sheet_by_name('Sheet1')
for i in range(sheet.nrows):
    data[sheet.row_values(i)[0]] = sheet.row_values(i)[1]

crop_root = "../front/src/assets/data/Seals"
crop_list = os.listdir(crop_root)

line = []
row = ['seal_name', 'seal_path', 'seal_interpretation', 'processed_name', 'processed_path', 'image_name', 'image_path',
       'front_page']
line.append(row)

cls = []

for file in crop_list:
    row = []
    name = file.split('_')[0] + ".jpg"
    if name.split('(')[0] not in cls:
        cls.append(name.split('(')[0])
        front_page = name
    row.append(file)
    row.append("@/assets/data/Seals/" + file)
    try:
        interpretation = data[file.split('.')[0]]
    except:
        interpretation = 'error-error-error'
    row.append(interpretation.split('-')[2])
    row.append(interpretation + '.jpg')
    row.append("@/assets/data/Processed/" + interpretation + '.jpg')
    row.append(name)
    row.append("@/assets/data/Images/" + name)
    row.append(front_page)
    line.append(row)

with open('correspond.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(line)
