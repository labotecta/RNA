cd C:\Articulos\IA\OCRaRNA\OCRaRNA
start "Digitos_DN_1_2" "C:\Articulos\IA\KRkb\KRkb\envTFsinGPU 3.8\Scripts\python.exe" RNA.py ^
--dird=F:/Articulos/IA/Datos/ ^
--dirr=F:/Articulos/IA/OCR/Digitos_DN_1_2/ ^
--fientrena=mnist_train ^
--fiprueba=mnist_test ^
--fisorpresa=mnist_test_azar_25 ^
--tipoRNA=0 ^
--objetos=0 ^
--referente=0 ^
--capas_ocultas=1 ^
--neuronas=2 ^
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
