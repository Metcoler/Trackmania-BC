import zipfile

with zipfile.ZipFile('get_data_driver.op', 'w') as z:
    z.write('info.toml', 'info.toml')
    z.write('main.as', 'main.as')