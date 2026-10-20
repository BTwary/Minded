using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

namespace Minded.Installer
{
    internal static class ReleaseInfo
    {
        public const string Version = "1.0.0";
    }

    public class SetupForm : Form
    {
        private Panel headerPanel;
        private Label titleLabel;
        private Label subtitleLabel;
        private Label welcomeLabel;
        private Label pathLabel;
        private TextBox pathTextBox;
        private Button browseButton;
        private CheckBox desktopShortcutCheck;
        private CheckBox startMenuShortcutCheck;
        private CheckBox launchAfterCheck;
        private ProgressBar progressBar;
        private Label statusLabel;
        private Button installButton;
        private Button cancelButton;
        private bool isFinished = false;

        public SetupForm()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.Text = "Minded — Autonomous Data Analyst Setup";
            this.Size = new Size(580, 470);
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(248, 249, 250);
            this.Font = new Font("Segoe UI", 9F, FontStyle.Regular);

            // 1. Header Banner
            headerPanel = new Panel
            {
                Dock = DockStyle.Top,
                Height = 85,
                BackColor = Color.FromArgb(5, 21, 10) // Deep Jungle
            };

            titleLabel = new Label
            {
                Text = "⚡ M I N D E D",
                Font = new Font("Segoe UI", 16F, FontStyle.Bold),
                ForeColor = Color.FromArgb(212, 175, 55), // Gold
                Location = new Point(24, 15),
                AutoSize = true
            };

            subtitleLabel = new Label
            {
                Text = "Autonomous Analytical Intelligence OS — Enterprise Desktop Setup",
                Font = new Font("Segoe UI", 9.5F, FontStyle.Regular),
                ForeColor = Color.FromArgb(247, 247, 232), // Ivory
                Location = new Point(26, 48),
                AutoSize = true
            };

            headerPanel.Controls.Add(titleLabel);
            headerPanel.Controls.Add(subtitleLabel);

            // 2. Welcome Description
            welcomeLabel = new Label
            {
                Text = "Welcome to the Minded Setup Wizard.\n\nThis installer will set up the autonomous data analyst platform on your system. It automatically acquires and expands required runtime components so Minded is ready to use immediately.",
                Location = new Point(28, 102),
                Size = new Size(510, 60),
                ForeColor = Color.FromArgb(33, 37, 41)
            };

            // 3. Destination Directory
            pathLabel = new Label
            {
                Text = "Installation Directory:",
                Location = new Point(28, 170),
                AutoSize = true,
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };

            string defaultPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Minded\AAOS");
            pathTextBox = new TextBox
            {
                Text = defaultPath,
                Location = new Point(28, 192),
                Size = new Size(410, 25),
                ReadOnly = true
            };

            browseButton = new Button
            {
                Text = "Browse...",
                Location = new Point(448, 191),
                Size = new Size(90, 27),
                BackColor = Color.FromArgb(233, 236, 239),
                FlatStyle = FlatStyle.Flat
            };
            browseButton.Click += (s, e) =>
            {
                using (FolderBrowserDialog fbd = new FolderBrowserDialog())
                {
                    fbd.Description = "Select Installation Directory for Minded";
                    fbd.SelectedPath = pathTextBox.Text;
                    if (fbd.ShowDialog() == DialogResult.OK)
                    {
                        pathTextBox.Text = fbd.SelectedPath;
                    }
                }
            };

            // 4. Options
            desktopShortcutCheck = new CheckBox
            {
                Text = "Create Desktop shortcut",
                Checked = true,
                Location = new Point(28, 230),
                AutoSize = true
            };

            startMenuShortcutCheck = new CheckBox
            {
                Text = "Create Start Menu shortcut",
                Checked = true,
                Location = new Point(28, 255),
                AutoSize = true
            };

            launchAfterCheck = new CheckBox
            {
                Text = "Launch Minded immediately after installation",
                Checked = true,
                Location = new Point(28, 280),
                AutoSize = true
            };

            // 5. Progress Area
            statusLabel = new Label
            {
                Text = "Ready to install.",
                Location = new Point(28, 320),
                Size = new Size(510, 20),
                ForeColor = Color.FromArgb(108, 117, 125)
            };

            progressBar = new ProgressBar
            {
                Location = new Point(28, 345),
                Size = new Size(510, 22),
                Visible = false,
                Minimum = 0,
                Maximum = 100
            };

            // 6. Action Buttons
            installButton = new Button
            {
                Text = "Install",
                Location = new Point(345, 385),
                Size = new Size(95, 32),
                BackColor = Color.FromArgb(40, 167, 69),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };
            installButton.Click += InstallButton_Click;

            cancelButton = new Button
            {
                Text = "Cancel",
                Location = new Point(448, 385),
                Size = new Size(90, 32),
                BackColor = Color.FromArgb(233, 236, 239),
                FlatStyle = FlatStyle.Flat
            };
            cancelButton.Click += (s, e) => this.Close();

            // Assemble Form
            this.Controls.Add(headerPanel);
            this.Controls.Add(welcomeLabel);
            this.Controls.Add(pathLabel);
            this.Controls.Add(pathTextBox);
            this.Controls.Add(browseButton);
            this.Controls.Add(desktopShortcutCheck);
            this.Controls.Add(startMenuShortcutCheck);
            this.Controls.Add(launchAfterCheck);
            this.Controls.Add(progressBar);
            this.Controls.Add(statusLabel);
            this.Controls.Add(installButton);
            this.Controls.Add(cancelButton);
        }

        private void InstallButton_Click(object sender, EventArgs e)
        {
            if (isFinished)
            {
                if (launchAfterCheck.Checked)
                {
                    string targetDir = Path.Combine(pathTextBox.Text, "app");
                    string exePath = Path.Combine(targetDir, "Minded.exe");
                    if (File.Exists(exePath))
                    {
                        Process.Start(new ProcessStartInfo(exePath) { WorkingDirectory = targetDir });
                    }
                }
                this.Close();
                return;
            }

            installButton.Enabled = false;
            cancelButton.Enabled = false;
            browseButton.Enabled = false;
            progressBar.Visible = true;
            statusLabel.Text = "Starting installation...";

            string installRoot = pathTextBox.Text;
            string appDir = Path.Combine(installRoot, "app");
            bool createDesktop = desktopShortcutCheck.Checked;
            bool createStartMenu = startMenuShortcutCheck.Checked;

            ThreadPool.QueueUserWorkItem(state =>
            {
                try
                {
                    PerformInstall(installRoot, appDir, createDesktop, createStartMenu);
                    this.Invoke((Action)(() =>
                    {
                        progressBar.Value = 100;
                        statusLabel.Text = "Installation completed successfully!";
                        statusLabel.ForeColor = Color.FromArgb(40, 167, 69);
                        installButton.Text = "Finish";
                        installButton.Enabled = true;
                        cancelButton.Visible = false;
                        isFinished = true;
                    }));
                }
                catch (Exception ex)
                {
                    this.Invoke((Action)(() =>
                    {
                        MessageBox.Show("Installation error: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        installButton.Enabled = true;
                        cancelButton.Enabled = true;
                        statusLabel.Text = "Installation failed.";
                    }));
                }
            });
        }

        private static bool HasValidWebView2Version(RegistryKey key)
        {
            if (key == null) return false;
            string value = key.GetValue("pv") as string;
            Version version;
            return Version.TryParse(value, out version) && version > new Version(0, 0, 0, 0);
        }

        private static bool IsWebView2RuntimeInstalled()
        {
            const string clientId = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";
            string[] machinePaths =
            {
                @"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\" + clientId,
                @"SOFTWARE\Microsoft\EdgeUpdate\Clients\" + clientId
            };

            foreach (string path in machinePaths)
            {
                using (RegistryKey key = Registry.LocalMachine.OpenSubKey(path))
                {
                    if (HasValidWebView2Version(key)) return true;
                }
            }

            using (RegistryKey key = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\EdgeUpdate\Clients\" + clientId))
            {
                if (HasValidWebView2Version(key)) return true;
            }

            string[] runtimeRoots =
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), @"Microsoft\EdgeWebView\Application"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Microsoft\EdgeWebView\Application"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\EdgeWebView\Application")
            };
            foreach (string runtimeRoot in runtimeRoots)
            {
                if (string.IsNullOrEmpty(runtimeRoot) || !Directory.Exists(runtimeRoot)) continue;
                try
                {
                    foreach (string versionDirectory in Directory.GetDirectories(runtimeRoot))
                    {
                        Version version;
                        if (Version.TryParse(Path.GetFileName(versionDirectory), out version) &&
                            File.Exists(Path.Combine(versionDirectory, "msedgewebview2.exe")))
                            return true;
                    }
                }
                catch { }
            }

            return false;
        }

        private static void EnsureWebView2Runtime(Assembly assembly)
        {
            if (IsWebView2RuntimeInstalled()) return;

            string installerPath = Path.Combine(Path.GetTempPath(), "MicrosoftEdgeWebview2Setup.exe");
            bool foundEmbedded = false;
            using (Stream resource = assembly.GetManifestResourceStream("MicrosoftEdgeWebView2RuntimeInstallerX64.exe"))
            {
                if (resource != null)
                {
                    using (FileStream output = new FileStream(installerPath, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        resource.CopyTo(output);
                    }
                    foundEmbedded = true;
                }
            }

            if (!foundEmbedded)
            {
                try
                {
                    ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072; // TLS 1.2
                    using (WebClient wc = new WebClient())
                    {
                        wc.DownloadFile("https://go.microsoft.com/fwlink/p/?LinkId=2124703", installerPath);
                    }
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException("WebView2 Runtime download failed: " + ex.Message);
                }
            }

            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = installerPath,
                    Arguments = "/silent /install",
                    WorkingDirectory = Path.GetDirectoryName(installerPath),
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using (Process process = Process.Start(startInfo))
                {
                    if (process != null) process.WaitForExit();
                }

                for (int attempt = 0; attempt < 30; attempt++)
                {
                    if (IsWebView2RuntimeInstalled()) return;
                    Thread.Sleep(1000);
                }
            }
            finally
            {
                try { if (File.Exists(installerPath)) File.Delete(installerPath); } catch { }
            }
        }

        private void PerformInstall(string installRoot, string appDir, bool createDesktop, bool createStartMenu)
        {
            Assembly asm = Assembly.GetExecutingAssembly();

            this.Invoke((Action)(() =>
            {
                statusLabel.Text = "Checking Microsoft Edge WebView2 Runtime...";
                progressBar.Value = 5;
            }));
            EnsureWebView2Runtime(asm);

            Directory.CreateDirectory(appDir);

            string tempZip = Path.Combine(Path.GetTempPath(), "MindedAAOS_payload_" + Guid.NewGuid().ToString("N") + ".zip");
            bool tempZipCreated = false;

            try
            {
                // Check if embedded in installer
                using (Stream zipStream = asm.GetManifestResourceStream("payload.zip"))
                {
                    if (zipStream != null)
                    {
                        using (FileStream fs = new FileStream(tempZip, FileMode.Create, FileAccess.Write))
                        {
                            zipStream.CopyTo(fs);
                        }
                        tempZipCreated = true;
                    }
                }

                // If not embedded, check if zip sits beside the installer
                if (!tempZipCreated)
                {
                    string currentDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                    string localZip1 = Path.Combine(currentDir, "payload.zip");
                    string localZip2 = Path.Combine(currentDir, "MindedAAOS-Windows-x64.zip");
                    string localZip = File.Exists(localZip1) ? localZip1 : (File.Exists(localZip2) ? localZip2 : null);

                    if (localZip != null)
                    {
                        tempZip = localZip;
                    }
                    else
                    {
                        // Download release bundle from GitHub Releases
                        string downloadUrl = "https://github.com/BTwary/Minded/releases/latest/download/MindedAAOS-Windows-x64.zip";
                        this.Invoke((Action)(() =>
                        {
                            statusLabel.Text = "Connecting to component server...";
                            progressBar.Value = 10;
                        }));

                        ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072; // TLS 1.2
                        using (WebClient client = new WebClient())
                        {
                            client.Headers.Add("User-Agent", "Minded-Bootstrapper/1.0");
                            client.DownloadProgressChanged += (s, ev) =>
                            {
                                this.Invoke((Action)(() =>
                                {
                                    int val = 10 + (int)(ev.ProgressPercentage * 0.45);
                                    progressBar.Value = Math.Min(55, Math.Max(10, val));
                                    statusLabel.Text = string.Format("Downloading components: {0:F1} MB / {1:F1} MB ({2}%)",
                                        ev.BytesReceived / (1024.0 * 1024.0),
                                        ev.TotalBytesToReceive > 0 ? ev.TotalBytesToReceive / (1024.0 * 1024.0) : 0,
                                        ev.ProgressPercentage);
                                }));
                            };

                            client.DownloadFileTaskAsync(new Uri(downloadUrl), tempZip).Wait();
                            tempZipCreated = true;
                        }
                    }
                }

                // Extract / Expand the ZIP archive
                this.Invoke((Action)(() =>
                {
                    statusLabel.Text = "Expanding application components...";
                    progressBar.Value = 55;
                }));

                using (ZipArchive archive = ZipFile.OpenRead(tempZip))
                {
                    int total = archive.Entries.Count;
                    int count = 0;
                    foreach (ZipArchiveEntry entry in archive.Entries)
                    {
                        string destPath = Path.Combine(appDir, entry.FullName);
                        if (string.IsNullOrEmpty(entry.Name))
                        {
                            Directory.CreateDirectory(destPath);
                        }
                        else
                        {
                            Directory.CreateDirectory(Path.GetDirectoryName(destPath));
                            entry.ExtractToFile(destPath, true);
                        }
                        count++;
                        int pct = 55 + (int)((count / (float)total) * 30);
                        if (count % 20 == 0 || count == total)
                        {
                            this.Invoke((Action)(() =>
                            {
                                progressBar.Value = Math.Min(85, pct);
                                statusLabel.Text = "Expanding: " + entry.Name;
                            }));
                        }
                    }
                }
            }
            finally
            {
                if (tempZipCreated && File.Exists(tempZip))
                {
                    try { File.Delete(tempZip); } catch { }
                }
            }

            // Extract launcher
            string launcherExe = Path.Combine(appDir, "Minded.exe");
            using (Stream exeStream = asm.GetManifestResourceStream("Minded.exe"))
            {
                if (exeStream != null)
                {
                    using (FileStream fs = new FileStream(launcherExe, FileMode.Create, FileAccess.Write))
                    {
                        exeStream.CopyTo(fs);
                    }
                }
            }

            // Extract uninstaller
            string uninstallerExe = Path.Combine(installRoot, "Uninstall.exe");
            using (Stream uninstStream = asm.GetManifestResourceStream("Uninstall.exe"))
            {
                if (uninstStream != null)
                {
                    using (FileStream fs = new FileStream(uninstallerExe, FileMode.Create, FileAccess.Write))
                    {
                        uninstStream.CopyTo(fs);
                    }
                }
            }

            // Offline marker
            File.WriteAllText(Path.Combine(appDir, "OFFLINE_MODE"), "1");

            this.Invoke((Action)(() => { progressBar.Value = 90; statusLabel.Text = "Creating shortcuts..."; }));

            // Shortcuts
            Type shellType = Type.GetTypeFromProgID("WScript.Shell");
            if (shellType != null)
            {
                dynamic shell = Activator.CreateInstance(shellType);

                if (createDesktop)
                {
                    string desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                    string lnk = Path.Combine(desktop, "Minded — Autonomous Data Analyst.lnk");
                    var sc = shell.CreateShortcut(lnk);
                    sc.TargetPath = launcherExe;
                    sc.WorkingDirectory = appDir;
                    sc.Description = "Minded — Autonomous Analytical Intelligence OS";
                    sc.Save();
                }

                if (createStartMenu)
                {
                    string startMenu = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Minded");
                    Directory.CreateDirectory(startMenu);

                    string appLnk = Path.Combine(startMenu, "Minded.lnk");
                    var sc = shell.CreateShortcut(appLnk);
                    sc.TargetPath = launcherExe;
                    sc.WorkingDirectory = appDir;
                    sc.Description = "Minded — Autonomous Analytical Intelligence OS";
                    sc.Save();

                    string uninstLnk = Path.Combine(startMenu, "Uninstall Minded.lnk");
                    var usc = shell.CreateShortcut(uninstLnk);
                    usc.TargetPath = uninstallerExe;
                    usc.WorkingDirectory = installRoot;
                    usc.Description = "Uninstall Minded";
                    usc.Save();
                }
            }

            // Register in Add/Remove Programs
            try
            {
                using (RegistryKey key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\MindedAAOS"))
                {
                    if (key != null)
                    {
                        key.SetValue("DisplayName", "Minded — Autonomous Analytical Intelligence OS");
                        key.SetValue("DisplayVersion", ReleaseInfo.Version);
                        key.SetValue("Publisher", "Minded");
                        key.SetValue("InstallLocation", installRoot);
                        key.SetValue("UninstallString", "\"" + uninstallerExe + "\"");
                        key.SetValue("NoModify", 1, RegistryValueKind.DWord);
                        key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
                    }
                }
            }
            catch { }
        }

        [STAThread]
        public static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SetupForm());
        }
    }
}
