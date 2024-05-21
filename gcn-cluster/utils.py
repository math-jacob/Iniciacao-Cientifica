from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook

def export_to_excel(df, filename):

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