
namespace Formas
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.label1 = new System.Windows.Forms.Label();
            this.ancho = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.alto = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.circulos = new System.Windows.Forms.TextBox();
            this.cuadrados = new System.Windows.Forms.TextBox();
            this.triangulos = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.fichero = new System.Windows.Forms.Label();
            this.sel = new System.Windows.Forms.Button();
            this.label7 = new System.Windows.Forms.Label();
            this.generar = new System.Windows.Forms.Button();
            this.muestra = new System.Windows.Forms.PictureBox();
            this.semilla = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.gira_triangulos = new System.Windows.Forms.CheckBox();
            this.gira_cuadrados = new System.Windows.Forms.CheckBox();
            this.mostrar_imagen = new System.Windows.Forms.CheckBox();
            this.ruido = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.cuenta = new System.Windows.Forms.Label();
            this.ruidoporciento = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.muestra)).BeginInit();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 42);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(134, 20);
            this.label1.TabIndex = 0;
            this.label1.Text = "Dimensión imagen";
            // 
            // ancho
            // 
            this.ancho.Location = new System.Drawing.Point(197, 38);
            this.ancho.Name = "ancho";
            this.ancho.Size = new System.Drawing.Size(46, 27);
            this.ancho.TabIndex = 1;
            this.ancho.Text = "28";
            this.ancho.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(249, 42);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(18, 20);
            this.label2.TabIndex = 2;
            this.label2.Text = "X";
            // 
            // alto
            // 
            this.alto.Location = new System.Drawing.Point(273, 38);
            this.alto.Name = "alto";
            this.alto.Size = new System.Drawing.Size(46, 27);
            this.alto.TabIndex = 3;
            this.alto.Text = "28";
            this.alto.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(13, 85);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(61, 20);
            this.label3.TabIndex = 4;
            this.label3.Text = "Círculos";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(13, 115);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(80, 20);
            this.label4.TabIndex = 5;
            this.label4.Text = "Cuadrados";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(13, 144);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(77, 20);
            this.label5.TabIndex = 6;
            this.label5.Text = "Triangulos";
            // 
            // circulos
            // 
            this.circulos.Location = new System.Drawing.Point(197, 83);
            this.circulos.Name = "circulos";
            this.circulos.Size = new System.Drawing.Size(46, 27);
            this.circulos.TabIndex = 7;
            this.circulos.Text = "10000";
            this.circulos.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // cuadrados
            // 
            this.cuadrados.Location = new System.Drawing.Point(197, 113);
            this.cuadrados.Name = "cuadrados";
            this.cuadrados.Size = new System.Drawing.Size(46, 27);
            this.cuadrados.TabIndex = 8;
            this.cuadrados.Text = "10000";
            this.cuadrados.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // triangulos
            // 
            this.triangulos.Location = new System.Drawing.Point(197, 142);
            this.triangulos.Name = "triangulos";
            this.triangulos.Size = new System.Drawing.Size(46, 27);
            this.triangulos.TabIndex = 9;
            this.triangulos.Text = "10000";
            this.triangulos.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(13, 221);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(121, 20);
            this.label6.TabIndex = 10;
            this.label6.Text = "Fichero de salida";
            // 
            // fichero
            // 
            this.fichero.BackColor = System.Drawing.Color.White;
            this.fichero.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.fichero.Location = new System.Drawing.Point(197, 221);
            this.fichero.Name = "fichero";
            this.fichero.Size = new System.Drawing.Size(532, 24);
            this.fichero.TabIndex = 11;
            // 
            // sel
            // 
            this.sel.Location = new System.Drawing.Point(735, 221);
            this.sel.Name = "sel";
            this.sel.Size = new System.Drawing.Size(39, 28);
            this.sel.TabIndex = 12;
            this.sel.Text = "Sel";
            this.sel.UseVisualStyleBackColor = true;
            this.sel.Click += new System.EventHandler(this.Sel_Click);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(13, 270);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(80, 20);
            this.label7.TabIndex = 13;
            this.label7.Text = "Generados";
            // 
            // generar
            // 
            this.generar.Location = new System.Drawing.Point(606, 267);
            this.generar.Name = "generar";
            this.generar.Size = new System.Drawing.Size(123, 28);
            this.generar.TabIndex = 15;
            this.generar.Text = "Generar";
            this.generar.UseVisualStyleBackColor = true;
            this.generar.Click += new System.EventHandler(this.Generar_Click);
            // 
            // muestra
            // 
            this.muestra.Location = new System.Drawing.Point(622, 42);
            this.muestra.Name = "muestra";
            this.muestra.Size = new System.Drawing.Size(152, 134);
            this.muestra.TabIndex = 16;
            this.muestra.TabStop = false;
            // 
            // semilla
            // 
            this.semilla.Location = new System.Drawing.Point(453, 38);
            this.semilla.Name = "semilla";
            this.semilla.Size = new System.Drawing.Size(91, 27);
            this.semilla.TabIndex = 18;
            this.semilla.Text = "11335";
            this.semilla.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(343, 42);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(90, 20);
            this.label8.TabIndex = 17;
            this.label8.Text = "Semilla azar";
            // 
            // gira_triangulos
            // 
            this.gira_triangulos.AutoSize = true;
            this.gira_triangulos.Location = new System.Drawing.Point(271, 142);
            this.gira_triangulos.Name = "gira_triangulos";
            this.gira_triangulos.Size = new System.Drawing.Size(63, 24);
            this.gira_triangulos.TabIndex = 20;
            this.gira_triangulos.Text = "Girar";
            this.gira_triangulos.UseVisualStyleBackColor = true;
            // 
            // gira_cuadrados
            // 
            this.gira_cuadrados.AutoSize = true;
            this.gira_cuadrados.Location = new System.Drawing.Point(271, 113);
            this.gira_cuadrados.Name = "gira_cuadrados";
            this.gira_cuadrados.Size = new System.Drawing.Size(63, 24);
            this.gira_cuadrados.TabIndex = 21;
            this.gira_cuadrados.Text = "Girar";
            this.gira_cuadrados.UseVisualStyleBackColor = true;
            // 
            // mostrar_imagen
            // 
            this.mostrar_imagen.AutoSize = true;
            this.mostrar_imagen.Checked = true;
            this.mostrar_imagen.CheckState = System.Windows.Forms.CheckState.Checked;
            this.mostrar_imagen.Location = new System.Drawing.Point(622, 12);
            this.mostrar_imagen.Name = "mostrar_imagen";
            this.mostrar_imagen.Size = new System.Drawing.Size(82, 24);
            this.mostrar_imagen.TabIndex = 22;
            this.mostrar_imagen.Text = "Mostrar";
            this.mostrar_imagen.UseVisualStyleBackColor = true;
            // 
            // ruido
            // 
            this.ruido.Location = new System.Drawing.Point(197, 172);
            this.ruido.Name = "ruido";
            this.ruido.Size = new System.Drawing.Size(46, 27);
            this.ruido.TabIndex = 24;
            this.ruido.Text = "10000";
            this.ruido.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(13, 174);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(48, 20);
            this.label9.TabIndex = 23;
            this.label9.Text = "Ruido";
            // 
            // cuenta
            // 
            this.cuenta.BackColor = System.Drawing.Color.White;
            this.cuenta.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.cuenta.Location = new System.Drawing.Point(129, 270);
            this.cuenta.Name = "cuenta";
            this.cuenta.Size = new System.Drawing.Size(97, 24);
            this.cuenta.TabIndex = 25;
            // 
            // ruidoporciento
            // 
            this.ruidoporciento.Location = new System.Drawing.Point(271, 172);
            this.ruidoporciento.Name = "ruidoporciento";
            this.ruidoporciento.Size = new System.Drawing.Size(46, 27);
            this.ruidoporciento.TabIndex = 26;
            this.ruidoporciento.Text = "25";
            this.ruidoporciento.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(323, 176);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(21, 20);
            this.label10.TabIndex = 27;
            this.label10.Text = "%";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 314);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.ruidoporciento);
            this.Controls.Add(this.cuenta);
            this.Controls.Add(this.ruido);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.mostrar_imagen);
            this.Controls.Add(this.gira_cuadrados);
            this.Controls.Add(this.gira_triangulos);
            this.Controls.Add(this.semilla);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.muestra);
            this.Controls.Add(this.generar);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.sel);
            this.Controls.Add(this.fichero);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.triangulos);
            this.Controls.Add(this.cuadrados);
            this.Controls.Add(this.circulos);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.alto);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.ancho);
            this.Controls.Add(this.label1);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.muestra)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox ancho;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox alto;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox circulos;
        private System.Windows.Forms.TextBox cuadrados;
        private System.Windows.Forms.TextBox triangulos;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label fichero;
        private System.Windows.Forms.Button sel;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Button generar;
        private System.Windows.Forms.PictureBox muestra;
        private System.Windows.Forms.TextBox semilla;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.CheckBox gira_triangulos;
        private System.Windows.Forms.CheckBox gira_cuadrados;
        private System.Windows.Forms.CheckBox mostrar_imagen;
        private System.Windows.Forms.TextBox ruido;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label cuenta;
        private System.Windows.Forms.TextBox ruidoporciento;
        private System.Windows.Forms.Label label10;
    }
}

