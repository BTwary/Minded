using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;
using Microsoft.Win32;

namespace Minded.Uninstall
{
    class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            DialogResult confirm = MessageBox.Show(
                "Are you sure you want to completely uninstall Minded — Autonomous Analytical Intelligence OS?",
                "Minded Uninstall",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question
            );

            if (confirm != DialogResult.Yes) return;

            try
            {
                // Stop every process of the installed product: the launcher (Minded.exe), the
                // packaged application (MindedAAOS.exe) and the PyInstaller-named executable.
                // Isolated analytical workers re-run the same executable, so they match too.
                string[] productProcesses = { "Minded", "MindedAAOS", "MindEd_AAOS_v1.0.0_Windows_x64" };
                string stillRunning = StopProductProcesses(productProcesses, 15000);
                if (stillRunning != null)
                {
                    MessageBox.Show(
                        "Minded is still running (" + stillRunning + ") and could not be closed.\n\n" +
                        "Close it from Task Manager and run the uninstaller again. Nothing was removed.",
                        "Uninstall Incomplete",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning
                    );
                    return;
                }

                // 1. Remove Desktop Shortcut
                string desktop = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                string lnk = Path.Combine(desktop, "Minded — Autonomous Data Analyst.lnk");
                if (File.Exists(lnk)) File.Delete(lnk);

                // 2. Remove Start Menu folder
                string startMenu = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Minded");
                if (Directory.Exists(startMenu)) Directory.Delete(startMenu, true);

                // 3. Remove Registry entry
                try
                {
                    Registry.CurrentUser.DeleteSubKeyTree(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\MindedAAOS", false);
                }
                catch { }

                // 4. Clean up app directory (via scheduled self-delete cmd)
                string localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                string appDir = Path.Combine(localApp, @"Minded\AAOS\app");

                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = "/C timeout /t 2 /nobreak > NUL & rmdir /s /q \"" + appDir + "\"",
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden
                };
                Process.Start(psi);

                MessageBox.Show(
                    "Minded has been successfully uninstalled from your computer.",
                    "Uninstall Complete",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information
                );
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error during uninstall: " + ex.Message, "Uninstall Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        // Returns null when every matching process has exited, otherwise the name of one that has not.
        static string StopProductProcesses(string[] names, int timeoutMs)
        {
            int selfId = Process.GetCurrentProcess().Id;
            DateTime deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
            string remaining = null;
            do
            {
                remaining = null;
                foreach (string name in names)
                {
                    foreach (Process p in Process.GetProcessesByName(name))
                    {
                        if (p.Id == selfId) continue;
                        try
                        {
                            if (!p.HasExited)
                            {
                                p.Kill();
                                p.WaitForExit(2000);
                            }
                            if (!p.HasExited) remaining = name;
                        }
                        catch (Exception)
                        {
                            // Access denied or already gone: re-check on the next pass.
                            try { if (!p.HasExited) remaining = name; } catch (Exception) { }
                        }
                    }
                }
            } while (remaining != null && DateTime.UtcNow < deadline);
            return remaining;
        }
    }
}
