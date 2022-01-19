from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox,filedialog
from ttkbootstrap import Style
from numpy import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mysql.connector
import seaborn as sns
import random
import operator
from operator import add, sub
import csv
import os

i = []
mydata= []

def handle_click(event):
   if trv.identify_region(event.x, event.y) == "separator":
       return "break"
   
def update(rows):
        global mydata
        mydata = rows
        trv.delete(*trv.get_children())
        for i in rows:
            trv.insert('', 'end', values=i)
        
def clrBox():
    t1.set("")
    t2.set("")
    t3.set("")
    t4.set("")
    t5.set("")
    t6.set("")
    t7.set("")

def search():
    q2 = q.get()
    query = "SELECT upisni_broj, smjer, bodovi_srednja,bodovi_matura,bodovi_suma, G1, G2, G3 FROM student WHERE smjer LIKE '%"+q2+"%' OR upisni_broj LIKE '%"+q2+"%' OR bodovi_srednja LIKE '%"+q2+"%' OR bodovi_matura LIKE '%"+q2+"%' OR bodovi_suma LIKE '%"+q2+"%'"
    cursor.execute(query)
    rows = cursor.fetchall()
    update(rows)  

def clear():
    query = "SELECT upisni_broj, smjer, bodovi_srednja,bodovi_matura,bodovi_suma, G1,G2,G3 FROM student"
    cursor.execute(query)
    rows = cursor.fetchall()
    update(rows)
    
def getrow(event):
    rowid = trv.identify_row(event.y)
    item = trv.item(trv.focus())
    #t1.set(item['values'][0])
    t2.set(item['values'][0])
    t3.set(item['values'][1])
    t4.set(item['values'][2])
    t5.set(item['values'][3])
    t6.set(item['values'][4])
    t7.set(item['values'][5])
    
def update_student():
    upisni_br = t2.get()
    smjer = t3.get()
    bod_srednja = t4.get()
    bod_matura = t5.get()
    bod_suma = t6.get()
    prva_god = t7.get()
    
    if messagebox.askyesno("Potvrda izmjene","Da li ste sigurni da želite izmjeniti podatke o studentu?"):
        query = "UPDATE student SET upisni_broj=%s,smjer=%s,bodovi_srednja=%s,bodovi_matura=%s,bodovi_suma=%s,G1=%s WHERE upisni_broj=%s AND smjer=%s"
        record=(upisni_br, smjer, bod_srednja, bod_matura, bod_suma, prva_god, upisni_br, smjer)
        cursor.execute(query, record)
        #mydb.commit()
        clear()
    else:
        return True

def add_new():
    upisni_br = t2.get()
    smjer = t3.get()
    bod_srednja = t4.get()
    bod_matura = t5.get()
    bod_suma = t6.get()
    prva_god = t7.get()
    query = "INSERT INTO student(id, upisni_broj, smjer, bodovi_srednja, bodovi_matura, bodovi_suma, G1) VALUES(NULL,%s,%s,%s,%s,%s,%s)"
    record=(upisni_br, smjer, bod_srednja, bod_matura, bod_suma, prva_god)
    try:
        cursor.execute(query, record)
    except mysql.connector.Error as err:
        print("Error: {}".format(err))
    #mydb.commit()
    clear()
    messagebox.showinfo("Student dodan","Željeni podaci su spremljeni u bazu.")

def delete_student():
    upisni_br = t2.get()
    smjer = t3.get()
    bod_srednja = t4.get()
    bod_matura = t5.get()
    bod_suma = t6.get()
    prva_god = t7.get()
    if upisni_br == "" or smjer == "" or bod_srednja == "" or bod_matura == "" or bod_suma == "" or prva_god == "":
        messagebox.showinfo("Nema odabira","Nije odabran student za brisanje ili je neki od podataka izbrisan iz čelije.")
    elif messagebox.askyesno("Potvrda brisanja", "Da li ste sigurni da želite izbrisati ovoga studenta?"):
        query = "DELETE FROM student WHERE upisni_broj=%s AND smjer=%s"
        cursor.execute(query, (upisni_br, smjer))
        #mydb.commit()
        clear()
    else:
        return True
    
def export():
    if len(mydata) <1:
        messagebox.showerror("Prazno", "Nema podataka za izvoz")
        return False
    
    fln = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Spremi CSV", filetypes=[("CSV File","*.csv")], defaultextension="*.*")
    with open(fln,mode='w',newline='') as myfile:
          exp_writer= csv.writer(myfile, delimiter=',')
          
          # dodavanje headera
          values = ("Upisni broj","Smjer","Bodovi srednja","Bodovi matura","Bodovi ukupno","G1", "G2", "G3")
          exp_writer.writerow(values)
          for i in mydata:
              exp_writer.writerow(i)
              
    messagebox.showinfo("Izvezeno", "Vaši podatci su izvezeni u datoteku: "+os.path.basename(fln)+".")          

def importcsv():
    mydata.clear()
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Otvori CSV", filetypes=[("CSV File","*.csv"),("All Files", "*.*")])
    with open(fln) as myfile:
        csvread = csv.reader(myfile,delimiter=",")
        skip = next(csvread)
        for i in csvread:
            mydata.append(i)
    update(mydata)        
                  
def savedb():
    if len(mydata) <1:
        messagebox.showerror("Prazno", "Nema podataka.")
        return False
    if messagebox.askyesno("Potvrda","Jeste li sigurni da želite spremiti podatke u bazu?"):
        for i in mydata:
            uBr = i[0]
            smj = i[1]
            bSr = i[2]
            bMa = i[3]
            bSum = i[4]
            prvaGod = i[5]
            drugaGod = None
            trecaGod = None
            
            if len(mydata[0]) > 6:
                drugaGod = i[6]
                trecaGod = i[7]
            
            if drugaGod is None or trecaGod is None:
                query= "INSERT INTO student (id, upisni_broj,smjer,bodovi_srednja,bodovi_matura,bodovi_suma, G1) VALUES(NULL, %s, %s, %s, %s, %s, %s)"
                try:
                    cursor.execute(query,(uBr,smj,bSr,bMa,bSum,prvaGod))
                except mysql.connector.Error as err:
                    print("Error: {}".format(err))
                
            else:
                query= "INSERT INTO student (id, upisni_broj,smjer,bodovi_srednja,bodovi_matura,bodovi_suma, G1, G2, G3) VALUES(NULL, %s, %s, %s, %s, %s, %s, %s, %s)"
                try:
                    cursor.execute(query,(uBr,smj,bSr,bMa,bSum,prvaGod,drugaGod,trecaGod))
                except mysql.connector.Error as err:
                    print("Error: {}".format(err))
                       
        #mydb.commit()
        clear()
        messagebox.showinfo("Spremljeno","Željeni podaci su spremljeni u bazu.")
    else:
        return False

def openPredikcija():
    if(len(mydata[0])<6):
        messagebox.showerror("Nepotpuno", "Nisu uneseni svi potrebni podatci za predikciju.")
        return False 
    elif len(mydata) <1:
        messagebox.showerror("Prazno", "Nema podataka za predikciju.")
        return False 
    else:
        win2 = Toplevel(root)
        
        # Centriranje prozora na sredinu ekrana    
        w = 650 # širina prozora
        h = 680 # visina prozora
    
        # uzimanje širine i visine ekrana
        ws = win2.winfo_screenwidth() # širina ekrana
        hs = win2.winfo_screenheight() # visina ekrana
    
        # računanje x i y koordinate za pozicioniranje prozora
        x1 = (ws/2) - (w/2)
        y1 = (hs/2) - (h/2)
    
        # dimenzije prozora i gdje je postavljen
        win2.geometry('%dx%d+%d+%d' % (w, h, x1, y1))
        win2.title("Predikcija")
    
        wrapper1 = LabelFrame(win2, text=(' Predikcija '))
        wrapper1.pack(fill='both', expand='yes',padx=5,pady=5)
        wrapper2 = LabelFrame(win2, text=(' Bodovi od 0 - 100 '))
        wrapper2.pack(fill='y', expand='yes',padx=5,pady=5)
        
        lbl1 = Label(wrapper2, text='Više od 88 ocjena je 5')
        lbl1.grid(row=0, column=0,padx=5,pady=3)
        lbl2 = Label(wrapper2, text='87-81 ocjena je 4')
        lbl2.grid(row=1, column=0,padx=5,pady=3)
        lbl3 = Label(wrapper2, text='80-67 ocjena je 3')
        lbl3.grid(row=2, column=0,padx=5,pady=3)
        lbl4 = Label(wrapper2, text='66-51 ocjena je 2')
        lbl4.grid(row=3, column=0,padx=5,pady=3)
        lbl5 = Label(wrapper2, text='Manje od 51 ocjena je 1')
        lbl5.grid(row=4, column=0,padx=5,pady=3)
        
        
        trv = ttk.Treeview(wrapper1, columns=(1,2,3,4,5,6,7,8), show='headings', height=20)
        trv.pack(side=LEFT, padx=5, pady=5)
        
        trv.heading(1, text='Upisni broj')
        trv.column(1, minwidth=100, width=100)
        trv.heading(2, text='Smjer')
        trv.column(2, minwidth=100, width=100)
        trv.heading(3, text='Bodovi srednja')
        trv.column(3, minwidth=100, width=100)
        trv.heading(4, text='Bodovi matura')
        trv.column(4, minwidth=100, width=100)
        trv.heading(5, text='Bodovi ukupno')
        trv.column(5, minwidth=100, width=100)
        trv.heading(6, text='G1')
        trv.column(6, minwidth=35, width= 35)
        trv.heading(7, text='G2')
        trv.column(7, minwidth=35,width= 35)
        trv.heading(8, text='G3')
        trv.column(8, minwidth=35,width= 35, stretch=False)
        
        #Vertikalna traka za pomicanje
        yscrollbar = ttk.Scrollbar(wrapper1, orient="vertical", command=trv.yview)
        yscrollbar.pack(side=RIGHT, fill="y",pady=10)
        trv.configure(yscrollcommand=yscrollbar.set)
        
        # PREDIKCIJA
        query = "SELECT id, upisni_broj, smjer, bodovi_srednja,bodovi_matura,bodovi_suma, G1,G2,G3 FROM student LIMIT 741"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        values = ("id",
          "Upisni broj",
          "Smjer",
          "Bodovi srednja",
          "Bodovi matura",
          "Bodovi ukupno",
          "G1",
          "G2",
          "G3"
          )
        
        df1= pd.DataFrame(rows, columns=values)
        
        df2 = pd.read_csv("upisni+ocjene.csv")
        df2 = df2.loc[:, df2.columns != 'Upisni broj']
        
        zbroj = pd.DataFrame((df2.sum(axis=1)/50)*100, columns = ["Finalni bodovi"])
        lista = zbroj.to_numpy().tolist()
        
        #DRUGA GODINA
        data = df1[['Bodovi srednja','Bodovi matura', 'Bodovi ukupno', 'G1', 'G2', 'G3']]
        predict = "G2"
        data = data.astype('int64')
        x = np.array(data.drop([predict], 1))
        y = np.array(data[predict])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        regr = LinearRegression()
        regr.fit(x_train, y_train)
        
        #Predikcija testnih podataka
        predictions = regr.predict(x_test)
        # for i in range(len(predictions)):
        #     print(predictions[i], x_test[i], [y_test[i]])
            
        upBr = []
        smjerovi = []
        bodSko = []
        bodMat = []
        bodUk = []
        ocjenePred1 = []
        ocjenePred2 = []
        
        br = 0
        while br < len(mydata):
            upBr.append(mydata[br][0])
            smjerovi.append(mydata[br][1])
            bodSko.append(mydata[br][2])
            bodMat.append(mydata[br][3])
            bodUk.append(mydata[br][4])
            ocjenePred1.append(float(mydata[br][5])) 
            if len(mydata[br])>=7:
                ocjenePred2.append(regr.predict([[mydata[br][2],mydata[br][3],mydata[br][4],mydata[br][5],mydata[br][7]]]))
            else: 
                predict = "G2"
                st = pd.DataFrame(mydata, columns=["Upisni broj", "Smjer", "Bodovi srednja", "Bodovi matura", "Bodovi ukupno", "G1"])
                st = st.drop('Upisni broj', axis=1)
                st = st.drop('Smjer', axis=1)
                
                st['Bodovi srednja'] = st['Bodovi srednja'].astype(float)
                st['Bodovi matura'] = st['Bodovi matura'].astype(float)
                st['Bodovi ukupno'] = st['Bodovi ukupno'].astype(float)
                st['G1'] = st['G1'].astype(float)
                result = pd.concat([data,st])
                result = result.drop("G3",axis=1)
                regr1 = LinearRegression()
                
                testdf = result[result['G2'].isnull()==True]
                traindf = result[result['G2'].isnull()==False]
                y = traindf['G2']
                traindf = traindf.drop("G2",axis=1)
                regr1.fit(traindf,y)
                testdf = testdf.drop("G2",axis=1)
                pred = regr1.predict(testdf)
                testdf['G2']= pred
                # print(br+1,". student")
                # print("Predviđeni broj bodova u 2.godini:", int(pred[br])) 
                # print(st.loc[br])
                ocjenePred2.append([pred[br]])

    
            br = br+1
        

        
        z = 0
        length = len(ocjenePred1)
        while z < length:
            if ocjenePred1[z] >= 88:
                ocjenePred1[z] = 5
            elif ocjenePred1[z] < 88 and ocjenePred1[z] >= 81:
                ocjenePred1[z] = 4
            elif ocjenePred1[z] < 81 and ocjenePred1[z] >= 67:
                ocjenePred1[z] = 3
            elif ocjenePred1[z] < 67 and ocjenePred1[z] >= 51:
                ocjenePred1[z] = 2
            else:
                ocjenePred1[z] = 1
            z = z+1  
            
        #TRECA GODINA
        data = df1[['Bodovi srednja','Bodovi matura', 'Bodovi ukupno', 'G1', 'G2', 'G3']]
        predict = "G3"
        data = data.astype('float64')
        x = np.array(data.drop([predict], 1))
        y = np.array(data[predict])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        regr = LinearRegression()
        regr.fit(x_train, y_train)

        prec = round((regr.score(x_test, y_test)*100),2)
        
        lbl6 = Label(wrapper2, text=("Preciznost:", prec,"%"))
        lbl6.grid(row=5, column=0,padx=5,pady=3)
        
        ocjenePred3 = []
        
        if(len(mydata[0]) >=7):
            br = 0
            while br < len(mydata):
                ocjenePred3.append(regr.predict([[mydata[br][2],mydata[br][3],mydata[br][4],mydata[br][5],mydata[br][6]]]))
                br = br+1
        else:
            br = 0
            while br < len(mydata):
                ocjenePred3.append(regr.predict([[float(mydata[br][2]),float(mydata[br][3]),float(mydata[br][4]),float(mydata[br][5]),float(ocjenePred2[br][0])]]))
                br = br+1
          
        #Dodavanje vrijednostima kako bi se dobila razlika u bodovima
        dodatak = [1, 2, 3, 4, 5, 6, 7, 8]

        operators = [operator.add, operator.add, operator.sub]
        random_operator = random.choice(operators)
        
        df = pd.DataFrame(np.random.choice(dodatak,size=(len(ocjenePred2), 2),p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05]), columns=['G2','G3'])
        
        ocjenePred2 = np.concatenate(ocjenePred2, axis=0)
        ocjenePred2 = ocjenePred2.tolist()

        ocjenePred3 = np.concatenate(ocjenePred3, axis=0)
        ocjenePred3 = ocjenePred3.tolist()
        
        operators = [operator.add, operator.add, operator.sub]
        random_operator = random.choice(operators)
        
        ocjenePred2 = pd.DataFrame(random_operator(ocjenePred2, df["G2"]))
        ocjenePred3 = pd.DataFrame(random_operator(ocjenePred3, df["G3"]))
        
        ocjenePred2 = ocjenePred2.to_numpy()
        ocjenePred3 = ocjenePred3.to_numpy()
        ocjenePred2 = [float(i) for i in ocjenePred2]
        ocjenePred3 = [float(i) for i in ocjenePred3]
        
        z = 0
        length = len(ocjenePred2)
        
        while z < length:
            if ocjenePred2[z] >= 88:
                ocjenePred2[z] = 5
            elif ocjenePred2[z] < 88 and ocjenePred2[z] >= 81:
                ocjenePred2[z] = 4
            elif ocjenePred2[z] < 81 and ocjenePred2[z] >= 70:
                ocjenePred2[z] = 3
            elif ocjenePred2[z] < 70 and ocjenePred2[z] >= 51:
                ocjenePred2[z] = 2
            else:
                ocjenePred2[z] = 1
            z = z+1  
            
        z = 0
        length = len(ocjenePred3)
        while z < length:
            if ocjenePred3[z] >= 88:
                ocjenePred3[z] = 5
            elif ocjenePred3[z] < 88 and ocjenePred3[z] >= 81:
                ocjenePred3[z] = 4
            elif ocjenePred3[z] < 81 and ocjenePred3[z] >= 67:
                ocjenePred3[z] = 3
            elif ocjenePred3[z] < 67 and ocjenePred3[z] >= 51:
                ocjenePred3[z] = 2
            else:
                ocjenePred3[z] = 1
            z = z+1  
        
        #ISPIS
        ispis = pd.DataFrame(columns = ['Upisni broj','Smjer', 'Bodovi srednja', 'Bodovi matura', 'Bodovi ukupno','G1','G2','G3'])
        ispis['Upisni broj'] = upBr 
        ispis['Smjer'] = smjerovi
        ispis['Bodovi srednja'] = bodSko
        ispis['Bodovi matura'] = bodMat
        ispis['Bodovi ukupno'] = bodUk
        ispis['G1'] = ocjenePred1
        ispis['G2'] = ocjenePred2
        ispis['G3'] = ocjenePred3
        ispis.loc[ispis.G2 > 100, 'G2'] = 100
        ispis.loc[ispis.G3 > 100, 'G3'] = 100
        predikcijaPrikaz = ispis.to_numpy().tolist()
        
        for row in predikcijaPrikaz:
         trv.insert("", "end", values=row)
         
        #prikaz grafa ocjena svih studenata upisanih u bazu
        ocjenePlot = ispis['G1'].value_counts().sort_index()
        ocjenePlot.plot.bar()
        plt.show()
        
        #prikaz heatmapa
        # data1 = pd.read_csv("ALL.csv")
        # data1 = data1[['Bodovi srednja','Bodovi matura','Bodovi ukupno', 'G1', 'G2', 'G3']]
        # plt.figure(figsize=(14,12))
        # sns.heatmap(data1.corr() ,annot=True)
        # plt.show()
        
mydb = mysql.connector.connect(host='localhost', 
                               user='root',
                               passwd='', 
                               database='predikcija_studenata',
                               auth_plugin='mysql_native_password')
cursor = mydb.cursor()

root = Tk()

#Stiliziranje
style = Style()
#tema
style.theme_use("flatly")

#Centriranje prozora na sredinu ekrana
w = 835 # širina prozora
h = 650 # visina prozora

# uzimanje širine i visine ekrana
ws = root.winfo_screenwidth() # širina ekrana
hs = root.winfo_screenheight() # visina ekrana

# računanje x i y koordinate za pozicioniranje prozora
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

# dimenzije prozora i gdje je postavljen
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.title('Predikcija uspjeha studenata')

#Varijable
q= StringVar()
t1= StringVar()
t2= StringVar()
t3= StringVar()
t4= StringVar()
t5= StringVar()
t6= StringVar()
t7= StringVar()

wrapper1 = LabelFrame(root, text=('Studenti'))
wrapper2 = LabelFrame(root, text=('Mogućnosti'))
wrapper3 = LabelFrame(root, text=('Podatci o odabranom studentu i predikcija'))
wrapper1.pack(fill='both', expand='yes',padx=15,pady=10)
wrapper2.pack(fill='both', expand='yes',padx=15,pady=10)
wrapper3.pack(fill='both', expand='yes',padx=15,pady=10)

trv = ttk.Treeview(wrapper1, columns=(2,3,4,5,6,7,8,9), show='headings', height=10)
trv.pack(side=LEFT, padx=5, pady=5)

#trv.heading(1, text='id')
#trv.column(1, minwidth=30, width=30)
trv.heading(2, text='Upisni broj')
trv.column(2, minwidth=100, width=100)
trv.heading(3, text='Smjer')
trv.column(3, minwidth=120, width=120)
trv.heading(4, text='Bodovi srednja')
trv.column(4, minwidth=100, width=100)
trv.heading(5, text='Bodovi matura')
trv.column(5, minwidth=100, width=100)
trv.heading(6, text='Bodovi ukupno')
trv.column(6, minwidth=100, width=100)
trv.heading(7, text='G1(0-100)')
trv.column(7, minwidth=80, width=80)
trv.heading(8, text='G2(0-100)')
trv.column(8, minwidth=80, width=80)
trv.heading(9, text='G3(0-100)')
trv.column(9, minwidth=80, width=80 ,stretch=NO)

#Dva puta lijevi klik na redak unutar treeview-a učitava podatke
trv.bind('<Double 1>', getrow)
trv.bind('<Button-1>', handle_click)

#Vertikalna traka za pomicanje
yscrollbar = ttk.Scrollbar(wrapper1, orient="vertical", command=trv.yview)
yscrollbar.pack(side=RIGHT, fill="y",pady=10)
trv.configure(yscrollcommand=yscrollbar.set)

#Gumbi unutar drugog okvira
expbtn = Button(wrapper2,text="Izvoz CSV", command = export)
expbtn.pack(side=tk.LEFT, padx=5, pady=5)
impbtn=Button(wrapper2,text="Uvoz CSV", command=importcsv)
impbtn.pack(side=tk.LEFT,padx=5,pady=5)
# savebtn = Button(wrapper2, text="Spremi u bazu", command=savedb)
# savebtn.pack(side=tk.LEFT,padx=5,pady=5)

#Inicijalizacija podataka unutar tablice
query = "SELECT upisni_broj, smjer, bodovi_srednja,bodovi_matura,bodovi_suma, G1, G2, G3 FROM student"
cursor.execute(query)
rows = cursor.fetchall()
update(rows)

#Tražilica
ent = Entry(wrapper2, textvariable=q)
ent.pack(side=tk.LEFT, padx=6)
btn = Button(wrapper2, text='Traži', command=search)
btn.pack(side=tk.LEFT,padx=6)
cbtn = Button(wrapper2,text='Prikaži sve', command=clear)
cbtn.pack(side=tk.LEFT,padx=6)

#Gumb za predikciju
btn_pred = Button(wrapper2,
             text ="Predikcija ocjena studenata",
             command = openPredikcija)
btn_pred.pack(side=tk.LEFT,padx=5,pady=5)

#Dio za korisničke podatke

# lbl1 = Label(wrapper3, text='id')
# lbl1.grid(row=0, column=0,padx=5,pady=3)
#ent1 = Entry(wrapper3, textvariable=t1, state=DISABLED)
#ent1.grid(row=0,column=1,padx=5,pady=3)

lbl2 = Label(wrapper3, text='Upisni broj')
lbl2.grid(row=0, column=0,padx=5,pady=3)
ent2 = Entry(wrapper3, textvariable=t2)
ent2.grid(row=0,column=1,padx=5,pady=3)
lbl3 = Label(wrapper3, text='Smjer')
lbl3.grid(row=1, column=0,padx=5,pady=3)
ent3 = Entry(wrapper3, textvariable=t3)
ent3.grid(row=1,column=1,padx=5,pady=3)
lbl4 = Label(wrapper3, text='Bodovi škola')
lbl4.grid(row=2, column=0,padx=5,pady=3)
ent4 = Entry(wrapper3, textvariable=t4)
ent4.grid(row=2,column=1,padx=5,pady=3)
lbl5 = Label(wrapper3, text='Bodovi matura')
lbl5.grid(row=3, column=0,padx=5,pady=3)
ent5 = Entry(wrapper3, textvariable=t5)
ent5.grid(row=3,column=1,padx=5,pady=3)
lbl6 = Label(wrapper3, text='Bodovi ukupno')
lbl6.grid(row=4, column=0,padx=5,pady=3)
ent6 = Entry(wrapper3, textvariable=t6)
ent6.grid(row=4,column=1,padx=5,pady=3)
lbl7 = Label(wrapper3, text='Prva godina(0-100)')
lbl7.grid(row=5, column=0,padx=5,pady=3)
ent7 = Entry(wrapper3, textvariable=t7)
ent7.grid(row=5,column=1,padx=5,pady=3)

#Gumbi za CRUD metode unutar tablice student Dodavanje/Izmjena/Brisanje
up_btn = Button(wrapper3,text='Promjeni',command=update_student)
add_btn = Button(wrapper3,text='Dodaj novog',command=add_new)
delete_btn = Button(wrapper3, text='Izbriši studenta', command=delete_student)

#Čišćenje polja 
clearBox = Button(wrapper3, text='Očisti polja', command=clrBox)

#Stavljanje gumbi u mrežu
# add_btn.grid(row=6, column=0, pady=3)
up_btn.grid(row=6, column=0, pady=3)
delete_btn.grid(row=6,column=1,pady=3)
clearBox.grid(row=6,column=2, pady=3)

root.mainloop()