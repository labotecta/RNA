cd C:\Articulos\IA\OCRaRNA\OCRaRNA
start "Formas_64x64_6000_DN_0_0_ref_2" "C:\Articulos\IA\KRkb\KRkb\envTFsinGPU 3.8\Scripts\python.exe" RNA.py ^
--dird=F:/Articulos/IA/Datos/ ^
--dirr=F:/Articulos/IA/OCR/Formas_64x64_6000_DN_0_0_r2/ ^
--fientrena=formas_train_6000_64x64 ^
--fiprueba=formas_test_30000_64x64 ^
--fisorpresa=formas_sorpresa_30000_64x64 ^
--objetos=1 ^
--referente=2 ^
--tipoRNA=0 ^
--capas_ocultas=0 ^
--neuronas=0 ^
--fil=32 ^
--mcv=5 ^
--ccv=2 ^
--scv=2 ^
--pases=50 ^
--vg=0 ^
--barridoini=0 ^
--montecarlo=3 ^
--sorteos 5000 5000 5000 ^
--pini 0 0 0 ^
--cortemc 0.99999 0.99999 0.99999 ^
--margenmc 4 4 4
