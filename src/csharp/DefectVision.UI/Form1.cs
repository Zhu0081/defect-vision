using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DefectVision.UI
{
    /// <summary>
    /// 第 5 课：参数可调（A）、缺陷轮廓（B）、批量检测（C）、历史记录（D）。
    /// 关键点：多边形坐标同样是归一化的，直接按比例乘图片宽高即可。
    /// </summary>
    public class Form1 : Form
    {
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        private const string BaseUrl = "http://127.0.0.1:8010";

        private Label lblTitle;
        private Label lblStatus;
        private Label lblResult;
        private Button btnConnect, btnPick, btnDetect;
        private Button btnBatch, btnExport, btnHistory;
        private PictureBox picBox;
        private TextBox txtLight, txtPart, txtConf;
        private ListBox lstHist;

        private Bitmap? _original;
        private string? _currentPath;
        private string _lastBatchDir = "";
        private readonly List<string> _batchCsv = new List<string>();

        public Form1()
        {
            Text = "工件表面缺陷检测 · 上位机";
            ClientSize = new Size(1240, 820);
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = Color.FromArgb(245, 247, 250);
            Font = new Font("Microsoft YaHei UI", 10F);

            lblTitle = new Label
            {
                Text = "工件表面缺陷检测",
                Font = new Font("Microsoft YaHei UI", 22F, FontStyle.Bold),
                AutoSize = true,
                Location = new Point(40, 24)
            };
            Controls.Add(lblTitle);

            lblStatus = new Label
            {
                Text = "状态：等待连接服务",
                ForeColor = Color.Gray,
                AutoSize = true,
                Location = new Point(44, 82)
            };
            Controls.Add(lblStatus);

            btnConnect = MakeButton("连接服务", 44, Color.FromArgb(16, 122, 106), Color.White);
            btnConnect.Click += BtnConnect_Click;
            Controls.Add(btnConnect);

            btnPick = MakeButton("选择图片", 206, Color.FromArgb(230, 235, 242), Color.Black);
            btnPick.Click += BtnPick_Click;
            Controls.Add(btnPick);

            btnDetect = MakeButton("开始检测", 368, Color.FromArgb(28, 92, 168), Color.White);
            btnDetect.Click += BtnDetect_Click;
            Controls.Add(btnDetect);

            picBox = new PictureBox
            {
                Location = new Point(44, 214),
                Size = new Size(820, 540),
                BackColor = Color.White,
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.Zoom
            };
            Controls.Add(picBox);

            lblResult = new Label
            {
                Text = "等待检测",
                Font = new Font("Microsoft YaHei UI", 13F, FontStyle.Bold),
                ForeColor = Color.Gray,
                AutoSize = false,
                Height = 44,
                Width = 820,
                Location = new Point(44, 764)
            };
            Controls.Add(lblResult);

            BuildRightPanel();

            string[] args = Environment.GetCommandLineArgs();
            if (args.Length > 1 && File.Exists(args[1]))
            {
                LoadImage(args[1]);
                if (args.Length > 2 && args[2] == "detect")
                {
                    Shown += (s2, e2) => BtnDetect_Click(null, EventArgs.Empty);
                }
            }
        }

        private void BuildRightPanel()
        {
            Label pTitle = new Label
            {
                Text = "检测参数",
                Font = new Font("Microsoft YaHei UI", 11F, FontStyle.Bold),
                Location = new Point(884, 214),
                AutoSize = true
            };
            Controls.Add(pTitle);

            AddParamRow("光源", 884, 246, out txtLight, "side");
            AddParamRow("件号", 884, 286, out txtPart, "022");
            AddParamRow("阈值", 884, 326, out txtConf, "0.25");

            btnBatch = new Button
            {
                Text = "批量检测…",
                Size = new Size(150, 40),
                Location = new Point(884, 372),
                BackColor = Color.FromArgb(230, 235, 242),
                FlatStyle = FlatStyle.Flat
            };
            btnBatch.Click += BtnBatch_Click;
            Controls.Add(btnBatch);

            btnExport = new Button
            {
                Text = "导出上次结果",
                Size = new Size(150, 40),
                Location = new Point(1044, 372),
                BackColor = Color.FromArgb(230, 235, 242),
                FlatStyle = FlatStyle.Flat
            };
            btnExport.Click += BtnExport_Click;
            Controls.Add(btnExport);

            btnHistory = new Button
            {
                Text = "刷新历史",
                Size = new Size(310, 36),
                Location = new Point(884, 424),
                BackColor = Color.FromArgb(230, 235, 242),
                FlatStyle = FlatStyle.Flat
            };
            btnHistory.Click += BtnHistory_Click;
            Controls.Add(btnHistory);

            lstHist = new ListBox
            {
                Location = new Point(884, 468),
                Size = new Size(310, 286),
                Font = new Font("Microsoft YaHei UI", 9F)
            };
            lstHist.SelectedIndexChanged += LstHist_SelectedIndexChanged;
            Controls.Add(lstHist);
        }

        private void AddParamRow(string label, int x, int y, out TextBox box, string def)
        {
            Label lb = new Label { Text = label, Location = new Point(x, y + 4), AutoSize = true };
            Controls.Add(lb);
            box = new TextBox { Location = new Point(x + 56, y), Size = new Size(90, 28), Text = def };
            Controls.Add(box);
        }

        private Button MakeButton(string text, int left, Color back, Color fore)
        {
            return new Button
            {
                Text = text,
                Size = new Size(150, 42),
                Location = new Point(left, 116),
                BackColor = back,
                ForeColor = fore,
                FlatStyle = FlatStyle.Flat
            };
        }

        // ---------------- 第 2 课：连接 ----------------
        private async void BtnConnect_Click(object? sender, EventArgs e)
        {
            btnConnect.Enabled = false;
            try
            {
                string json = await Http.GetStringAsync(BaseUrl + "/health");
                using (JsonDocument doc = JsonDocument.Parse(json))
                {
                    JsonElement root = doc.RootElement;
                    lblStatus.Text = $"状态：已连接  ·  数据库 {GetStr(root, "db")}  ·  模型 {GetStr(root, "model_name")}";
                    lblStatus.ForeColor = Color.FromArgb(16, 122, 106);
                }
            }
            catch (Exception ex)
            {
                lblStatus.Text = "状态：连接失败 —— " + ex.Message;
                lblStatus.ForeColor = Color.Crimson;
            }
            finally { btnConnect.Enabled = true; }
        }

        // ---------------- 第 3 课：选图 ----------------
        private void BtnPick_Click(object? sender, EventArgs e)
        {
            using (OpenFileDialog dlg = new OpenFileDialog())
            {
                dlg.Title = "选一张要检测的图片";
                dlg.Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp|所有文件|*.*";
                dlg.InitialDirectory = @"F:\defect-vision\data\yolo\images\test";
                if (dlg.ShowDialog(this) == DialogResult.OK) LoadImage(dlg.FileName);
            }
        }


        /// <summary>
        /// 按 EXIF 的 Orientation 标签把图片摆正。
        /// 关键：cv2.imread 会自动按 EXIF 旋转，而 .NET 的 Image.FromFile 不会。
        /// 两边看到的方向不一致，模型返回的坐标就对不上——这是最隐蔽的一类 bug。
        /// </summary>
        private static Bitmap LoadOriented(string path)
        {
            using (Image src = Image.FromFile(path))
            {
                Bitmap bmp = new Bitmap(src);
                try
                {
                    foreach (PropertyItem p in src.PropertyItems)
                    {
                        if (p.Id != 0x0112 || p.Value == null || p.Value.Length == 0) continue;
                        switch (p.Value[0])
                        {
                            case 2: bmp.RotateFlip(RotateFlipType.RotateNoneFlipX); break;
                            case 3: bmp.RotateFlip(RotateFlipType.Rotate180FlipNone); break;
                            case 4: bmp.RotateFlip(RotateFlipType.Rotate180FlipX); break;
                            case 5: bmp.RotateFlip(RotateFlipType.Rotate90FlipX); break;
                            case 6: bmp.RotateFlip(RotateFlipType.Rotate90FlipNone); break;
                            case 7: bmp.RotateFlip(RotateFlipType.Rotate270FlipX); break;
                            case 8: bmp.RotateFlip(RotateFlipType.Rotate270FlipNone); break;
                        }
                        break;
                    }
                }
                catch { /* 没有 EXIF 就用原图 */ }
                return bmp;
            }
        }

        private void LoadImage(string path)
        {
            try
            {
                _original = LoadOriented(path);
                _currentPath = path;
                SwapImage(new Bitmap(_original));   // 给控件一份副本：SwapImage 会 Dispose 旧图，不能把 _original 交出去
                lblStatus.Text = $"已载入：{Path.GetFileName(path)}   ({_original.Width} × {_original.Height})";
                lblStatus.ForeColor = Color.FromArgb(60, 60, 60);
                lblResult.Text = "已载入，点「开始检测」";
                lblResult.ForeColor = Color.Gray;
            }
            catch (Exception ex)
            {
                lblStatus.Text = "图片打不开 —— " + ex.Message;
                lblStatus.ForeColor = Color.Crimson;
            }
        }

        private void SwapImage(Bitmap bmp)
        {
            Image? old = picBox.Image;
            picBox.Image = bmp;
            old?.Dispose();
        }

        // ---------------- 第 4/5 课：检测 + 参数 + 轮廓 ----------------
        private async void BtnDetect_Click(object? sender, EventArgs e)
        {
            if (_original == null || _currentPath == null)
            {
                lblResult.Text = "先选一张图片";
                lblResult.ForeColor = Color.DarkOrange;
                return;
            }
            btnDetect.Enabled = false;
            lblResult.Text = "检测中…";
            lblResult.ForeColor = Color.Gray;
            try
            {
                JsonElement root = await DetectFileAsync(_currentPath);
                SwapImage(DrawResult(_original, root));
                ShowVerdict(root);
            }
            catch (Exception ex)
            {
                lblResult.Text = "检测失败 —— " + ex.Message;
                lblResult.ForeColor = Color.Crimson;
            }
            finally { btnDetect.Enabled = true; }
        }

        private async Task<JsonElement> DetectFileAsync(string path)
        {
            using (MultipartFormDataContent form = new MultipartFormDataContent())
            using (FileStream fs = File.OpenRead(path))
            {
                form.Add(new StreamContent(fs), "file", Path.GetFileName(path));
                string url = string.Format("{0}/detect?light={1}&part={2}&conf={3}",
                    BaseUrl,
                    Uri.EscapeDataString(txtLight.Text.Trim()),
                    Uri.EscapeDataString(txtPart.Text.Trim()),
                    Uri.EscapeDataString(txtConf.Text.Trim()));
                string json = await (await Http.PostAsync(url, form)).Content.ReadAsStringAsync();
                File.AppendAllText(@"F:\defect-vision\_debug_detect.json", json + "`r`n", new UTF8Encoding(false));
                using (JsonDocument doc = JsonDocument.Parse(json))
                {
                    return doc.RootElement.Clone();   // doc 释放后 Clone 仍然可用
                }
            }
        }

        private void ShowVerdict(JsonElement root)
        {
            string verdict = GetStr(root, "verdict");
            int n = root.GetProperty("n_defects").GetInt32();
            double maxScore = root.GetProperty("max_score").GetDouble();
            double ms = root.GetProperty("latency_ms").GetDouble();
            lblResult.Text = string.Format("{0}   ·   {1} 处缺陷   ·   最高 {2:0.000}   ·   {3:0} ms",
                verdict == "ng" ? "NG 不良" : "OK 良品", n, maxScore, ms);
            lblResult.ForeColor = verdict == "ng" ? Color.Crimson : Color.FromArgb(16, 122, 106);
        }

        /// <summary>先画轮廓（有的话），再画外接框和标签。</summary>
        private Bitmap DrawResult(Bitmap src, JsonElement root)
        {
            Bitmap bmp = new Bitmap(src);   // 1:1 复制像素，不让 GDI+ 做任何缩放
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.PageUnit = GraphicsUnit.Pixel;
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

                float penW = Math.Max(3f, src.Width / 500f);
                float fontSize = Math.Max(10f, src.Width / 160f);

                using (Font font = new Font("Microsoft YaHei UI", fontSize, FontStyle.Bold))
                using (SolidBrush fill = new SolidBrush(Color.FromArgb(70, 230, 30, 60)))
                using (Pen edge = new Pen(Color.FromArgb(230, 30, 60), penW))
                using (Pen boxPen = new Pen(Color.FromArgb(120, 230, 30, 60), Math.Max(2f, penW / 2)))
                {
                    if (!root.TryGetProperty("items", out JsonElement items) ||
                        items.ValueKind != JsonValueKind.Array) return bmp;

                    foreach (JsonElement it in items.EnumerateArray())
                    {
                        JsonElement box = it.GetProperty("box");
                        float x1 = (float)box[0].GetDouble() * src.Width;
                        float y1 = (float)box[1].GetDouble() * src.Height;
                        float x2 = (float)box[2].GetDouble() * src.Width;
                        float y2 = (float)box[3].GetDouble() * src.Height;

                        // 轮廓：多边形点集直接按比例换算
                        if (it.TryGetProperty("polygon", out JsonElement poly) &&
                            poly.ValueKind == JsonValueKind.Array && poly.GetArrayLength() >= 3)
                        {
                            PointF[] pts = new PointF[poly.GetArrayLength()];
                            for (int i = 0; i < pts.Length; i++)
                            {
                                pts[i] = new PointF(
                                    (float)poly[i][0].GetDouble() * src.Width,
                                    (float)poly[i][1].GetDouble() * src.Height);
                            }
                            g.FillPolygon(fill, pts);
                            g.DrawPolygon(edge, pts);
                        }
                        else
                        {
                            g.DrawRectangle(boxPen, x1, y1, x2 - x1, y2 - y1);
                        }

                        string tag = string.Format("{0} {1:0.00}", GetStr(it, "label"),
                            it.GetProperty("score").GetDouble());
                        SizeF sz = g.MeasureString(tag, font);
                        g.FillRectangle(Brushes.White, x1, Math.Max(0, y1 - sz.Height), sz.Width, sz.Height);
                        g.DrawString(tag, font, Brushes.Black, x1, Math.Max(0, y1 - sz.Height));
                    }
                }
            }
            return bmp;
        }

        // ---------------- C. 批量检测 ----------------
        private async void BtnBatch_Click(object? sender, EventArgs e)
        {
            using (FolderBrowserDialog dlg = new FolderBrowserDialog())
            {
                dlg.Description = "选一个文件夹，里面的图会逐张送去检测";
                dlg.SelectedPath = @"F:\defect-vision\data\yolo\images\test";
                if (dlg.ShowDialog(this) != DialogResult.OK) return;

                string[] files = Directory.GetFiles(dlg.SelectedPath, "*.*");
                List<string> images = new List<string>();
                foreach (string f in files)
                {
                    string ext = Path.GetExtension(f).ToLowerInvariant();
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                        images.Add(f);
                }
                if (images.Count == 0) { lblResult.Text = "这个文件夹里没有图片"; return; }

                _batchCsv.Clear();
                _lastBatchDir = dlg.SelectedPath;
                _batchCsv.Add("filename,verdict,n_defects,max_score,latency_ms,classes");

                btnBatch.Enabled = false;
                int ok = 0, done = 0;
                foreach (string f in images)
                {
                    done++;
                    lblResult.Text = string.Format("批量检测 {0}/{1} …", done, images.Count);
                    lblResult.ForeColor = Color.Gray;
                    Application.DoEvents();
                    try
                    {
                        JsonElement root = await DetectFileAsync(f);
                        string verdict = GetStr(root, "verdict");
                        int n = root.GetProperty("n_defects").GetInt32();
                        double ms2 = root.GetProperty("latency_ms").GetDouble();
                        double mx = root.GetProperty("max_score").GetDouble();
                        List<string> labels = new List<string>();
                        if (root.TryGetProperty("items", out JsonElement its))
                            foreach (JsonElement it in its.EnumerateArray())
                                labels.Add(GetStr(it, "label"));

                        _batchCsv.Add(string.Format("{0},{1},{2},{3},{4},{5}",
                            Path.GetFileName(f), verdict, n,
                            mx.ToString("0.000", CultureInfo.InvariantCulture),
                            ms2.ToString("0.0", CultureInfo.InvariantCulture),
                            string.Join("|", labels)));
                        if (verdict == "ng") ok++;
                    }
                    catch (Exception ex)
                    {
                        _batchCsv.Add(Path.GetFileName(f) + ",ERROR,0,0,0," + ex.Message.Replace(",", ";"));
                    }
                }
                btnBatch.Enabled = true;
                lblResult.Text = string.Format("批量完成：{0} 张，其中不良 {1} 张。点「导出上次结果」存 CSV",
                    images.Count, ok);
                lblResult.ForeColor = Color.FromArgb(28, 92, 168);
            }
        }

        private void BtnExport_Click(object? sender, EventArgs e)
        {
            if (_batchCsv.Count == 0)
            {
                lblResult.Text = "还没有批量结果，先点「批量检测」";
                lblResult.ForeColor = Color.DarkOrange;
                return;
            }
            string outPath = Path.Combine(_lastBatchDir, "batch_result.csv");
            File.WriteAllLines(outPath, _batchCsv, new UTF8Encoding(true));  // 带 BOM，Excel 才不乱码
            lblResult.Text = "已导出：" + outPath;
            lblResult.ForeColor = Color.FromArgb(16, 122, 106);
        }

        // ---------------- D. 历史记录 ----------------
        private async void BtnHistory_Click(object? sender, EventArgs e)
        {
            lstHist.Items.Clear();
            try
            {
                string json = await Http.GetStringAsync(BaseUrl + "/records?limit=20");
                using (JsonDocument doc = JsonDocument.Parse(json))
                {
                    foreach (JsonElement r in doc.RootElement.EnumerateArray())
                    {
                        string id = r.GetProperty("id").GetInt32().ToString();
                        string verdict = GetStr(r, "verdict");
                        string n = r.GetProperty("n_defects").GetInt32().ToString();
                        string created = GetStr(r, "created_at");
                        // 接口给的是带 +00:00 的 UTC 时间。原样截断显示的就是 UTC，会慢 8 小时，
                        // 所以先解析再转本地时区；解析不了就退回原来的截断写法。
                        string when = created;
                        if (DateTimeOffset.TryParse(created, CultureInfo.InvariantCulture,
                                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt))
                            when = dt.ToLocalTime().ToString("MM-dd HH:mm", CultureInfo.InvariantCulture);
                        else if (created.Length >= 19)
                            when = created.Substring(5, 14).Replace('T', ' ');
                        lstHist.Items.Add(string.Format("#{0}  {1}  {2}处  {3}",
                            id, verdict == "ng" ? "NG" : "OK", n, when));
                    }
                }
            }
            catch (Exception ex)
            {
                lblStatus.Text = "历史加载失败 —— " + ex.Message;
                lblStatus.ForeColor = Color.Crimson;
            }
        }

        private async void LstHist_SelectedIndexChanged(object? sender, EventArgs e)
        {
            if (lstHist.SelectedItem == null) return;
            string text = lstHist.SelectedItem.ToString() ?? "";
            if (!text.StartsWith("#")) return;
            int end = text.IndexOf(' ');
            if (end < 2) return;
            string idText = text.Substring(1, end - 1);
            try
            {
                byte[] bytes = await Http.GetByteArrayAsync(BaseUrl + "/image/" + idText);
                using (MemoryStream ms = new MemoryStream(bytes))
                using (Image img = Image.FromStream(ms))
                {
                    _original = new Bitmap(img);
                }
                SwapImage(_original);
                lblStatus.Text = "正在查看历史记录 " + text;
                lblStatus.ForeColor = Color.FromArgb(60, 60, 60);
                lblResult.Text = "这是历史记录里的原图（不含框）";
                lblResult.ForeColor = Color.Gray;
                _currentPath = null;
            }
            catch (Exception ex)
            {
                lblStatus.Text = "取历史图片失败 —— " + ex.Message;
                lblStatus.ForeColor = Color.Crimson;
            }
        }

        private static string GetStr(JsonElement root, string name)
        {
            if (root.TryGetProperty(name, out JsonElement v))
                return v.ValueKind == JsonValueKind.Null ? "null" : v.ToString();
            return "?";
        }
    }
}
