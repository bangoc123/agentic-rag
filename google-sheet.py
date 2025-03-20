import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

# Add credentials to the account
credentials = ServiceAccountCredentials.from_json_keyfile_name('mles-class-12c1216b7303.json', scope)

# Authorize the clientsheet
client = gspread.authorize(credentials)

# Get the sheet (by name or by URL)
# sheet = client.open('Your Sheet Name').sheet1  # For first sheet
sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/1mOkgLyo1oedOG1nlvoSHpqK9-fTFzE9ysLuKob9TXlg').sheet1



# Example operations
# Get all values
data = sheet.get_all_records()
print(data)

# Get specific cell
cell_value = sheet.cell(1, 1).value  # Row 1, Column 1
print(cell_value)

# Update a cell
sheet.update_cell(1, 1, "New Value")