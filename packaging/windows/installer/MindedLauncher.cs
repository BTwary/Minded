using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

namespace Minded.Desktop
{
    public class LauncherApplication : ApplicationContext
    {
        private Process _serverProcess;
        private Process _appWindowProcess;
        private NotifyIcon _trayIcon;
        private const int Port = 8000;
        private const string Host = "127.0.0.1";

        public LauncherApplication()
        {
            SetBrowserFeatureControl();
            InitializeTray();
            StartServerAndOpenAppWindow();
        }

        private void SetBrowserFeatureControl()
        {
            // Enable modern IE11 rendering mode for embedded WebBrowser fallback
            try
            {
                string appName = Path.GetFileName(Assembly.GetExecutingAssembly().Location);
                using (RegistryKey key = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Internet Explorer\Main\FeatureControl\FEATURE_BROWSER_EMULATION"))
                {
                    if (key != null)
                    {
                        key.SetValue(appName, 11001, RegistryValueKind.DWord);
                    }
                }
            }
            catch { }
        }

        private void InitializeTray()
        {
            ContextMenu menu = new ContextMenu();
            menu.MenuItems.Add("⚡ Open Minded Desktop Window", (s, e) => LaunchDesktopAppWindow("http://" + Host + ":" + Port + "/cockpit"));
            menu.MenuItems.Add("📊 Open Platform Overview", (s, e) => LaunchDesktopAppWindow("http://" + Host + ":" + Port + "/"));
            menu.MenuItems.Add("📁 Open Data Directory", (s, e) => OpenDataDir());
            menu.MenuItems.Add("-");
            menu.MenuItems.Add("🛑 Exit Minded", (s, e) => ExitApplication());

            _trayIcon = new NotifyIcon
            {
                Text = "Minded — Autonomous Analytical Intelligence OS",
                Icon = SystemIcons.Application,
                ContextMenu = menu,
                Visible = true
            };
            _trayIcon.DoubleClick += (s, e) => LaunchDesktopAppWindow("http://" + Host + ":" + Port + "/cockpit");
        }

        private void StartServerAndOpenAppWindow()
        {
            ThreadPool.QueueUserWorkItem(state =>
            {
                try
                {
                    if (!IsPortOpen(Host, Port))
                    {
                        string pythonPath = ResolvePythonPath();
                        if (string.IsNullOrEmpty(pythonPath))
                        {
                            MessageBox.Show(
                                "Could not locate a compatible Python runtime.\nPlease install Python 3.10+ or reinstall Minded using MindedSetup.exe.",
                                "Minded — Runtime Missing",
                                MessageBoxButtons.OK,
                                MessageBoxIcon.Error
                            );
                            ExitApplication();
                            return;
                        }

                        string appDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                        ProcessStartInfo psi = new ProcessStartInfo
                        {
                            FileName = pythonPath,
                            Arguments = "-m uvicorn apps.api.src.main:app --host " + Host + " --port " + Port,
                            WorkingDirectory = appDir,
                            CreateNoWindow = true,
                            UseShellExecute = false,
                            WindowStyle = ProcessWindowStyle.Hidden
                        };

                        psi.EnvironmentVariables["AAOS_OFFLINE_MODE"] = "1";
                        psi.EnvironmentVariables["AAOS_BUSINESS_TIMEZONE"] = "UTC";
                        psi.EnvironmentVariables["AI_ENABLED"] = "false";
                        psi.EnvironmentVariables["AI_PROVIDER"] = "none";
                        psi.EnvironmentVariables["TELEMETRY_ENABLED"] = "false";
                        psi.EnvironmentVariables["FEEDBACK_UPLOAD_ENABLED"] = "false";
                        psi.EnvironmentVariables["STORAGE_PROVIDER"] = "local";
                        psi.EnvironmentVariables["PYTHONPATH"] = appDir;

                        _serverProcess = Process.Start(psi);

                        // Wait for server to bind to port
                        int waited = 0;
                        while (!IsPortOpen(Host, Port) && waited < 20000)
                        {
                            Thread.Sleep(300);
                            waited += 300;
                            if (_serverProcess != null && _serverProcess.HasExited)
                            {
                                MessageBox.Show(
                                    "Minded backend failed to start. Exit code: " + _serverProcess.ExitCode,
                                    "Minded Startup Error",
                                    MessageBoxButtons.OK,
                                    MessageBoxIcon.Error
                                );
                                ExitApplication();
                                return;
                            }
                        }
                    }

                    // Open dedicated native desktop application window
                    Thread.Sleep(500);
                    LaunchDesktopAppWindow("http://" + Host + ":" + Port + "/cockpit");

                    if (_trayIcon != null)
                    {
                        _trayIcon.ShowBalloonTip(
                            2500,
                            "Minded is Running",
                            "Minded Desktop Window is active. Right-click this tray icon anytime for controls.",
                            ToolTipIcon.Info
                        );
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error starting Minded: " + ex.Message, "Minded Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            });
        }

        private static bool IsPortOpen(string host, int port)
        {
            try
            {
                using (TcpClient client = new TcpClient())
                {
                    IAsyncResult result = client.BeginConnect(host, port, null, null);
                    bool success = result.AsyncWaitHandle.WaitOne(400);
                    if (!success) return false;
                    client.EndConnect(result);
                    return true;
                }
            }
            catch
            {
                return false;
            }
        }

        private void LaunchDesktopAppWindow(string url)
        {
            try
            {
                string browserPath = ResolveAppWindowHost();
                string userDataDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    @"Minded\AAOS\window_profile"
                );
                if (!Directory.Exists(userDataDir))
                {
                    Directory.CreateDirectory(userDataDir);
                }

                if (!string.IsNullOrEmpty(browserPath) && File.Exists(browserPath))
                {
                    // If window is already running, focus or launch new instance
                    if (_appWindowProcess != null && !_appWindowProcess.HasExited)
                    {
                        // Launch window instance
                    }

                    ProcessStartInfo psi = new ProcessStartInfo
                    {
                        FileName = browserPath,
                        Arguments = string.Format(
                            "--app={0} --user-data-dir=\"{1}\" --window-size=1400,900 --disable-extensions --disable-features=Translate --app-id=MindedAAOS",
                            url,
                            userDataDir
                        ),
                        UseShellExecute = false
                    };

                    _appWindowProcess = Process.Start(psi);
                }
                else
                {
                    // Native Windows Forms Fallback Frame
                    LaunchNativeFormWindow(url);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not launch desktop window: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private static void LaunchNativeFormWindow(string url)
        {
            MethodInvoker action = delegate
            {
                Form form = new Form
                {
                    Text = "Minded — Autonomous Analytical Intelligence OS",
                    Size = new Size(1366, 850),
                    StartPosition = FormStartPosition.CenterScreen,
                    Icon = SystemIcons.Application
                };

                WebBrowser wb = new WebBrowser
                {
                    Dock = DockStyle.Fill,
                    IsWebBrowserContextMenuEnabled = true,
                    ScriptErrorsSuppressed = true
                };
                wb.Navigate(url);
                form.Controls.Add(wb);
                form.Show();
            };

            if (Application.OpenForms.Count > 0 && Application.OpenForms[0].InvokeRequired)
            {
                Application.OpenForms[0].BeginInvoke(action);
            }
            else
            {
                action();
            }
        }

        private static string ResolveAppWindowHost()
        {
            string[] candidates = {
                // Microsoft Edge (Standard Windows 10/11)
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), @"Microsoft\Edge\Application\msedge.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Microsoft\Edge\Application\msedge.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Microsoft\Edge\Application\msedge.exe"),
                // Google Chrome
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Google\Chrome\Application\chrome.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), @"Google\Chrome\Application\chrome.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Google\Chrome\Application\chrome.exe")
            };

            foreach (string p in candidates)
            {
                if (File.Exists(p)) return p;
            }

            // Check App Paths registry
            try
            {
                using (RegistryKey key = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe"))
                {
                    if (key != null)
                    {
                        object val = key.GetValue(null);
                        if (val != null && File.Exists(val.ToString())) return val.ToString();
                    }
                }
                using (RegistryKey key = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"))
                {
                    if (key != null)
                    {
                        object val = key.GetValue(null);
                        if (val != null && File.Exists(val.ToString())) return val.ToString();
                    }
                }
            }
            catch { }

            return null;
        }

        private static void OpenDataDir()
        {
            string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Minded\AAOS");
            if (Directory.Exists(path))
            {
                Process.Start("explorer.exe", path);
            }
        }

        private static string ResolvePythonPath()
        {
            string appDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            string[] relativeCandidates = {
                Path.Combine(appDir, @"runtime\python.exe"),
                Path.Combine(appDir, @"python\python.exe"),
                Path.Combine(appDir, @"_internal\python.exe"),
                Path.Combine(appDir, @".venv\Scripts\python.exe")
            };

            foreach (string p in relativeCandidates)
            {
                if (File.Exists(p)) return p;
            }

            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string[] localCandidates = {
                Path.Combine(localAppData, @"Python\pythoncore-3.14-64\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python314\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python313\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python312\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python311\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python310\python.exe")
            };

            foreach (string p in localCandidates)
            {
                if (File.Exists(p)) return p;
            }

            // Search PATH
            string pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
            foreach (string dir in pathEnv.Split(';'))
            {
                string clean = dir.Trim().Trim('"');
                if (string.IsNullOrEmpty(clean)) continue;
                string py = Path.Combine(clean, "python.exe");
                if (File.Exists(py)) return py;
            }

            return null;
        }

        private void ExitApplication()
        {
            if (_trayIcon != null)
            {
                _trayIcon.Visible = false;
                _trayIcon.Dispose();
            }

            if (_appWindowProcess != null && !_appWindowProcess.HasExited)
            {
                try
                {
                    _appWindowProcess.Kill();
                }
                catch { }
            }

            if (_serverProcess != null && !_serverProcess.HasExited)
            {
                try
                {
                    _serverProcess.Kill();
                }
                catch { }
            }

            ExitThread();
        }

        [STAThread]
        public static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LauncherApplication());
        }
    }
}
