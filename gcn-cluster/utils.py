from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
import csv

def export_to_excel(df, filename):
  filename += '.xlsx'
  # Supondo que 'df' seja seu DataFrame
  # Arredondando os valores da coluna 'accuracy' para 4 casas decimais
  df['accuracy'] = df['accuracy'].round(4)

  # Criando um novo arquivo Excel
  wb = Workbook()
  ws = wb.active

  # Adicionando os dados do DataFrame ao arquivo Excel
  for r_idx, row in enumerate(dataframe_to_rows(df, index=False), 1):
    for c_idx, value in enumerate(row, 1):
      if isinstance(value, float):
        value = '{:.4f}'.format(value).replace('.', ',')  # Formatar o valor com vírgula
      ws.cell(row=r_idx, column=c_idx, value=value)

  # Salvando o arquivo Excel
  wb.save(filename)
  
def dic_to_csv(dictionary, filename):
  filename += '.csv'
  with open(filename, "w", newline="") as file:
    w = csv.writer(file)
    keys = list(dictionary.keys())
    w.writerow(keys)
    
    max_len = max([len(dictionary[key]) for key in keys])

    for i in range(0,max_len):
      row = []
      for key in keys:
        if (i < len(dictionary[key])):  
          row.append(dictionary[key][i])
        else:
          row.append('')
      w.writerow(row)