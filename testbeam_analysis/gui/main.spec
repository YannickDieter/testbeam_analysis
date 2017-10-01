# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['/home/davidlp/git/testbeam_analysis/testbeam_analysis/gui'],
             binaries=[],
             datas=[ ('dut_types.yaml', '.') ],
             hiddenimports=['setuptools.msvc', 'pixel_clusterizer.cluster_functions', 'numpydoc', 'progressbar-latest'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='main',
          debug=False,
          strip=False,
          upx=True,
          console=True , resources=['analysis_functions.so,dll,analysis_functions.so'])
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='main')
