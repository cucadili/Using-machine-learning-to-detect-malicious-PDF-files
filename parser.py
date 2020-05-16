import os
import re
import csv
keywords = ("endobj","obj","endstream","stream",
            "startxref","xref", "trailer",
            "/Pages","/PageLayout","/PageLabels","/PageMode","/Page",
            "/FFilter","/Filter", "/FDecodeParams","/First","/F",
            "/Predictor","/Prev","/Pattern","/Perms  ","/PieceInfo","/Parent ","/P",
            "/ObjStm","/OpenAction","/Outlines","/O",
            "/RichMedia","/Rows","/Root","/Resources","/Rotate ","/R",
            "/ColorTransform","/Colors","/Color",
            "/EncryptMetadata","/Encrypt","/Kids","/K",
            "/URI","/URLS","/U",
            "/Names","/N","/Version","/VP","/V",
            "/JS","/JavaScript","/AA","/AcroForm","/JBIG2Decode",
            "/Launch","/EmbeddedFile","/XFA",
            "/Lenght","/DL","/DecodeParams",
            "/Columns","/EarlyChange","/BitsPerComponent",
            "/EndOfLine","/EncodedByAlign",
            "/Blackls1","/EndOfBlock","/DamagedRowsBeforeErrors","/JBIG2Globals",
            "/Size","/ID","/Info","/Type", "/Extends","/XRefStm","/Subfilter",
            "/CF","/StmF","/StrF","/EFF","/Dests","/Extensions",
            "/Lang","/Metadata","/SpiderInfo ",
            "/MarkInfo","/Legal","/Collection ","/NeedRendering",
            "/Count ","/LastModified",
            "/MediaBox","/CropBox ","/BleedBox ","/ArtBox",
            "/TrimBox","/PZ ","/Templates","/IDS","/XObject")
  
#анализ файла
def statistics(file):
    try:
        infile = open(file, 'rb')
        data=infile.read()
        infile.close()
        count=[]
        #print(file)
        #print(''.join([chr(byte) for byte in data]))
        str_dat=''.join([chr(byte) for byte in data])
        for word in keywords:
            words=len(re.findall(word, str_dat))
            if word == "obj":
                words-=count[0]
            elif word == "stream":
                words-=count[2]
            elif word =="/Page":
                words = words - count[7]-count[8]-count[9]-count[10]
            elif word == "xref":
                words-=count[4]
            elif word == "/F":
                words=words-count[12]-count[13]-count[14]-count[15]
            elif word == "/P":
                words = words - count[7]-count[8]-count[9]-count[10]-count[11]-count[17]-count[18]-count[19]-count[20]-count[21]-count[22]
            elif word =="/O":
                words = words - count[24]-count[25]-count[26]
            elif word=="/R":
                words = words - count[28]-count[29]-count[30]-count[31]-count[32]
            elif words =="/Color":
                words = words - count[34]-count[35]
            elif word =="/Encrypt":
                words-=count[37]
            elif words =="/K":
                words-=count[39]
            elif words =="/U":
                words=words-count[41]-count[42]
            elif words =="/N":
                words-=count[44]
            elif words =="/V":
                words=words-count[46]-count[47]
            count.append(words)
            #print(word,words)

    except:
        return []
    return count
    
#получаем список файлов
def analyze():
    path="D:\\files_5"
    csva="D:\\parse.csv"
    csvf="D:\\flag.csv"
    file=open(csva , "w" , newline = "")
    fileflag=open(csvf,"w",newline="")
    #with open(csva , "w" , newline = "") as file:
     #   with open(csvf,"w",newline="")as fileflag:
    wr=csv.writer(file,delimiter=';')
    wf=csv.writer(fileflag,delimiter=';')
    wr.writerow(keywords)
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.find('.pdf')!=-1:
                st=statistics(root+'\\'+filename)
                if st!=[]:
                    wr.writerow(st)
                    if root.find('malware')!=-1:
                        wf.writerow([1])
                    else:
                        wf.writerow([0])
                    print(root+'\\'+filename)
               

if __name__ == '__main__':
	analyze()
