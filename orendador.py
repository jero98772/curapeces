import os
import datetime
import time
import subprocess
#importamos  las libererias
hoy=datetime.datetime.now().strftime('%Y-%m-%d')
directorio= "modelos_de_inteligencia_artificial_variedad"
class contenido:
	def contenido():
#se crea la funcion y clase contenido 
		#contenido =input(" las palabras no pueden ir separado por espacios por que creera que es otro 'comando' ni simbolos \n tema para el contenido : \n")
		contenido="models curapeces"
		if contenido=="":
			contenido= "contenido"
#validar que el campo se llene "automaticamente"
		return contenido
class archivo_existe:
	def archivo_existe():
#se crea la funcion y clase archivo_existe
		numero_de_la_clase=1
		nombre ="curapeces"
#variables

		for en_dir in os.listdir(directorio):
#recorrro el diectorio
#el for convierte los archivos en elementos de una lista que se puden contar individualmente
			numero_carpetas=len(os.listdir(directorio))
#numero_carpetas divide en elemntos cuantativos los elemmentos del dierctorio o carpeta 
			print("catidad de carpetas",numero_carpetas)
			#if not  os.walk(en_dir) in os.listdir(directorio) :
#la condicion comemntada seria que se no se encuntra en la carpeta que se va crear para el control de el agoritmo   
			#while numero_carpetas > numero_carpetas:
#while "siempre" va ser una condicion verdera pero con esto impedimos que salga de control el algorimo 
			nombre_numero_y_asignatura=nombre+str(numero_carpetas)+"__"+str(contenido)+"__"+str(hoy)								
			print(nombre_numero_y_asignatura)
			direccion=directorio+"/"+nombre_numero_y_asignatura
			os.mkdir(direccion)
			return direccion
#craea y retorna la carpeta creada despues de haber sido nombrada

	
class setup:
	def carpeta_inical():
#se crea la funcion carpeta_inical y clase setup
		print(directorio)

		if not os.path.exists(directorio):
			os.mkdir(directorio)
# si la carpeta no exite cree la carpeta madre 
			numero_carpetas=len(os.listdir(directorio))
#numero_carpetas divide en elemntos cuantativos los elemmentos del dierctorio o carpeta 

			print(numero_carpetas)
			if numero_carpetas ==0:
				carpeta_vacia=str(numero_carpetas)+"no_borrar_por_el_flujo_del_algorimo"
				os.mkdir(directorio+"/"+carpeta_vacia)
#si carpetas es 0 cree un carpeta inicial que afecta el flujo del algoritmo
setup.carpeta_inical()
contenido=contenido.contenido()
"""
class 	crear_contenido:
	def crear_contenido():
#se crea la funcion y clase crear_contenido
		nombre=archivo_existe.archivo_existe()
		veces_apoyo=eval(input("cantidad de imagenes para tener apoyo visual\n"))
		for i  in range(veces_apoyo):
			comado_kolourpaint=str("touch "+str(nombre)+"/"+str(contenido)+"_dibujo"+str(i)+".jpg")
			subprocess.run(str(comado_kolourpaint) , shell=True)
		
		comado_org=str("touch "+str(nombre)+"/"+str(contenido)+"_notebook.org")
		subprocess.run(str(comado_org) , shell=True)

		comado_codigo=str("touch "+str(nombre)+"/"+str(contenido)+"_codigo.py")
		subprocess.run(str(comado_codigo) , shell=True)
#y ya mediante los comados se craea las cosas que se quieran y la cantidad que quieran y como se quieran  
crear_contenido.crear_contenido()
class 	crear_contenido:
	def crear_contenido():
#se crea la funcion y clase crear_contenido
		nombre=archivo_existe.archivo_existe()"""