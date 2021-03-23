using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Formas
{
    public partial class Form1 : Form
    {
        FileStream fw;
        StreamWriter sw;
        Random azar;

        public Form1()
        {
            InitializeComponent();
        }

        private void Sel_Click(object sender, EventArgs e)
        {
            SaveFileDialog ficheroescritura = new SaveFileDialog()
            {
                FileName = string.Empty,
                Filter = "CSV (*.csv)|*.csv|All files (*.*)|*.*",
                FilterIndex = 1,
            };
            //ficheroescritura.RestoreDirectory = ficheroescritura.OverwritePrompt = false;
            //ficheroescritura.CheckPathExists = true;
            if (ficheroescritura.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    fichero.Text = ficheroescritura.FileName;
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message, "Fichero de salida", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }
        private bool FicheroEscritura()
        {
            try
            {
                if (sw != null) sw.Close();
                string nbfichero = fichero.Text.Trim();
                fw = new FileStream(nbfichero, FileMode.Create, FileAccess.Write, FileShare.Read);
                sw = new StreamWriter(fw);
                return true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Fichero de salida", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            if (sw != null) sw.Close();
            return false;
        }
        private void Generar_Click(object sender, EventArgs e)
        {
            if (!FicheroEscritura()) return;
            try
            {
                int SEMILLA = Convert.ToInt32(semilla.Text.Trim());
                azar = new Random(SEMILLA);
                if (!GeneraFormas(Convert.ToInt32(circulos.Text.Trim()), 0, false)) return;
                sw.Flush();
                if (!GeneraFormas(Convert.ToInt32(cuadrados.Text.Trim()), 1, gira_cuadrados.Checked)) return;
                sw.Flush();
                if (!GeneraFormas(Convert.ToInt32(triangulos.Text.Trim()), 2, gira_triangulos.Checked)) return;
                sw.Flush();
                if (!GeneraFormas(Convert.ToInt32(ruido.Text.Trim()), 3, false)) return;
                sw.Close();
                MessageBox.Show("Terminado", "Genera Formas", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Generar formas", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        private bool GeneraFormas(int n, int tipo, bool girar)
        {
            try
            {
                int an = Convert.ToInt32(ancho.Text.Trim());
                int al = Convert.ToInt32(alto.Text.Trim());
                if (mostrar_imagen.Checked) muestra.Size = new System.Drawing.Size(an, al);
                Bitmap imagen = new Bitmap(an, al);
                Graphics g = Graphics.FromImage(imagen);
                Bitmap imagen_r = new Bitmap(an, al);
                Graphics gr = Graphics.FromImage(imagen_r);
                int centrox;
                int centroy;
                int radio;
                int rmax = Math.Min(an, al) / 2 - 2;
                double xmin, xmax, ymin, ymax;
                float angle;
                StringBuilder linea;
                Point[] tri;
                int i;
                int nr = 0;
                if (tipo == 3)
                {
                    nr = (int)(an * al * Convert.ToInt32(ruidoporciento.Text.Trim()) / 100.0);
                }
                int contador = 0;
                while (contador < n)
                {
                    linea = new StringBuilder();
                    radio = (int)(azar.NextDouble() * rmax);
                    if (radio < 4) continue;
                    xmin = radio;
                    xmax = an - radio;
                    ymin = radio;
                    ymax = al - radio;
                    centrox = (int)(xmin + azar.NextDouble() * (xmax - xmin));
                    centroy = (int)(ymin + azar.NextDouble() * (ymax - ymin));
                    g.Clear(Color.Black);
                    if (tipo == 0)
                    {
                        g.FillEllipse(Brushes.White, centrox - radio, centroy - radio, 2 * radio, 2 * radio);
                    }
                    else if (tipo == 1)
                    {
                        g.FillRectangle(Brushes.White, centrox - radio, centroy - radio, 2 * radio, 2 * radio);
                    }
                    else if (tipo == 2)
                    {
                        tri = new Point[] { new Point(centrox - radio, centroy + radio), new Point(centrox + radio, centroy + radio), new Point(centrox, centroy - radio) };
                        g.FillPolygon(Brushes.White, tri);
                    }
                    else
                    {
                        if (contador == 0)
                        {
                            // Sin nada

                            g.Clear(Color.Black);
                        }
                        else if (contador == 1)
                        {
                            // Todo blanco

                            g.Clear(Color.White);
                        }
                        else
                        {
                            i = 0;
                            while (i < nr)
                            {
                                centrox = (int)(azar.NextDouble() * an);
                                centroy = (int)(azar.NextDouble() * al);
                                if (imagen.GetPixel(centrox, centroy).R == 0)
                                {
                                    g.FillRectangle(Brushes.White, centrox, centroy, 1, 1);
                                    i++;
                                }
                            }
                        }
                    }
                    if (girar)
                    {
                        imagen_r = new Bitmap(an, al);
                        gr = Graphics.FromImage(imagen_r);
                        gr.TranslateTransform((float)centrox, (float)centroy);
                        angle = (float)(azar.NextDouble() * 180);
                        gr.RotateTransform(angle);
                        gr.TranslateTransform(-(float)centrox, -(float)centroy);
                        gr.DrawImage(imagen, new Point(0, 0));
                        g.Clear(Color.Black);
                        g.DrawImage(imagen_r, new Point(0, 0));
                    }
                    if (mostrar_imagen.Checked) muestra.Image = imagen;
                    linea.Append(string.Format("{0}", tipo));
                    for (int j = 0; j < al; j++)
                    {
                        for (int k = 0; k < an; k++)
                        {
                            linea.Append(string.Format(",{0}", imagen.GetPixel(k, j).R));
                        }
                    }
                    sw.WriteLine(linea.ToString());
                    contador++;
                    if (contador % 100 == 0)
                    {
                        cuenta.Text = string.Format("{0}", contador);
                        Application.DoEvents();
                    }
                    //MessageBox.Show("Pulsa", "Genera Circulos", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                return true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Genera Circulos", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return false;
            }
        }
    }
}
