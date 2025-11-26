import zipfile

with zipfile.ZipFile('C:\\Users\\sampu\\OpenplanetNext\\Plugins\\get_data_driver.op', 'w') as z:
    z.write('info.toml', 'info.toml')
    z.write('main.as', 'main.as')