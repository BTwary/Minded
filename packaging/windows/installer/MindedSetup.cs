using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

namespace Minded.Installer
{
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
            this.Size = new Size(580, 460);
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
                Text = "Welcome to the Minded Setup Wizard.\n\nThis installer will install the complete autonomous data analyst platform on your system. Minded runs 100% locally and offline without external cloud dependencies.",
                Location = new Point(28, 105),
                Size = new Size(510, 60),
                ForeColor = Color.FromArgb(33, 37, 41)
            };

            // 3. Destination Directory
            pathLabel = new Label
            {
                Text = "Installation Directory:",
                Location = new Point(28, 175),
                AutoSize = true,
                Font = new Font("Segoe UI", 9F, FontStyle.Bold)
            };

            string defaultPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Minded\AAOS");
            pathTextBox = new TextBox
            {
                Text = defaultPath,
                Location = new Point(28, 198),
                Size = new Size(410, 25),
                ReadOnly = true
            };

            browseButton = new Button
            {
                Text = "Browse...",
                Location = new Point(446, 197),
                Size = new Size(92, 27)
            };
            browseButton.Click += (s, e) =>
            {
                using (FolderBrowserDialog fbd = new FolderBrowserDialog())
                {
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
                Text = "Create Desktop Shortcut (Minded — Autonomous Data Analyst)",
                Location = new Point(32, 238),
                Size = new Size(500, 24),
                Checked = true
            };

            startMenuShortcutCheck = new CheckBox
            {
                Text = "Create Start Menu shortcut in Programs folder",
                Location = new Point(32, 264),
                Size = new Size(500, 24),
                Checked = true
            };

            launchAfterCheck = new CheckBox
            {
                Text = "Launch Minded immediately after installation",
                Location = new Point(32, 290),
                Size = new Size(500, 24),
                Checked = true
            };

            // 5. Progress Bar and Status
            progressBar = new ProgressBar
            {
                Location = new Point(28, 325),
                Size = new Size(510, 22),
                Visible = false,
                Style = ProgressBarStyle.Continuous
            };

            statusLabel = new Label
            {
                Text = "Ready to install.",
                Location = new Point(28, 352),
                Size = new Size(510, 20),
                ForeColor = Color.FromArgb(108, 117, 125)
            };

            // 6. Action Buttons
            installButton = new Button
            {
                Text = "Install Now",
                Location = new Point(334, 380),
                Size = new Size(100, 32),
                BackColor = Color.FromArgb(212, 175, 55),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                Cursor = Cursors.Hand
            };
            installButton.Click += InstallButton_Click;

            cancelButton = new Button
            {
                Text = "Cancel",
                Location = new Point(444, 380),
                Size = new Size(94, 32)
            };
            cancelButton.Click += (s, e) => this.Close();

            // Add Controls to Form
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
            statusLabel.Text = "Installing Minded files...";

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
                    if (key == null)
                        return false;

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
                            if (HasValidWebView2Version(key))
                                return true;
                        }
                    }

                    using (RegistryKey key = Registry.CurrentUser.OpenSubKey(
                        @"Software\Microsoft\EdgeUpdate\Clients\" + clientId))
                    {
                        if (HasValidWebView2Version(key))
                            return true;
                    }

                    string[] runtimeRoots =
                    {
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), @"Microsoft\EdgeWebView\Application"),
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Microsoft\EdgeWebView\Application"),
                        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\EdgeWebView\Application")
                    };
                    foreach (string runtimeRoot in runtimeRoots)
                    {
                        if (string.IsNullOrEmpty(runtimeRoot) || !Directory.Exists(runtimeRoot))
                            continue;
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
                        catch (IOException) { }
                        catch (UnauthorizedAccessException) { }
                    }

                    return false;
                }

                private static void EnsureWebView2Runtime(Assembly assembly)
                {
                    if (IsWebView2RuntimeInstalled())
                        return;

                    const string resourceName = "MicrosoftEdgeWebView2RuntimeInstallerX64.exe";
                    string installerPath = Path.Combine(Path.GetTempPath(), resourceName);

                    using (Stream resource = assembly.GetManifestResourceStream(resourceName))
                    {
                        if (resource == null)
                            throw new InvalidOperationException(
                                "The offline WebView2 Runtime installer is missing from this setup package.");

                        using (FileStream output = new FileStream(
                            installerPath, FileMode.Create, FileAccess.Write, FileShare.None))
                        {
                            resource.CopyTo(output);
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

                        int exitCode;
                        using (Process process = Process.Start(startInfo))
                        {
                            if (process == null)
                                throw new InvalidOperationException("Could not start the WebView2 Runtime installer.");
                            process.WaitForExit();
                            exitCode = process.ExitCode;
                        }

                        // Recheck detection; exit code alone does not establish runtime availability.
                        for (int attempt = 0; attempt < 30; attempt++)
                        {
                            if (IsWebView2RuntimeInstalled())
                                return;
                            Thread.Sleep(1000);
                        }

                        throw new InvalidOperationException(
                            "WebView2 Runtime installation was not detected after setup. Installer exit code: " +
                            exitCode + ". If setup requested a restart, restart Windows and run setup again.");
                    }
                    finally
                    {
                        try
                        {
                            if (File.Exists(installerPath))
                                File.Delete(installerPath);
                        }
                        catch (IOException) { }
                        catch (UnauthorizedAccessException) { }
                    }
                }

        private void PerformInstall(string installRoot, string appDir, bool createDesktop, bool createStartMenu)
        {
            Assembly asm = Assembly.GetExecutingAssembly();

            this.Invoke((Action)(() =>
            {
                statusLabel.Text = "Checking Microsoft Edge WebView2 Runtime...";
                progressBar.Value = Math.Min(progressBar.Maximum, 5);
            }));
            EnsureWebView2Runtime(asm);

            Directory.CreateDirectory(appDir);

            // 1. Extract payload.zip if embedded
            using (Stream zipStream = asm.GetManifestResourceStream("payload.zip"))
            {
                if (zipStream != null)
                {
                    using (ZipArchive archive = new ZipArchive(zipStream))
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
                            int pct = (int)((count / (float)total) * 70);
                            this.Invoke((Action)(() =>
                            {
                                progressBar.Value = Math.Min(70, pct);
                                statusLabel.Text = "Extracting: " + entry.Name;
                            }));
                        }
                    }
                }
            }

            if (!File.Exists(Path.Combine(appDir, "MindedAAOS.exe")) ||
                !Directory.Exists(Path.Combine(appDir, "_internal")) ||
                !File.Exists(Path.Combine(appDir, "frontend", "index.html")))
            {
                throw new InvalidDataException(
                    "The embedded application payload is incomplete or has an incorrect root layout. " +
                    "Required files: MindedAAOS.exe, _internal, and frontend/index.html.");
            }

            // 2. Extract Minded.exe launcher
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

            // 3. Extract Uninstall.exe
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

            // 4. Write offline markers & manifest
            File.WriteAllText(Path.Combine(appDir, "OFFLINE_MODE"), "1");
            string manifestJson = "{\n  \"offline\": true,\n  \"network_policy\": \"loopback-only\",\n  \"runtime\": \"self-contained\"\n}";
            File.WriteAllText(Path.Combine(appDir, "MindedAAOS.exe.manifest.json"), manifestJson);

            this.Invoke((Action)(() => { progressBar.Value = 85; statusLabel.Text = "Creating shortcuts..."; }));

            // 5. Create Shortcuts via COM
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

            // 6. Register in Windows Programs & Features
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
