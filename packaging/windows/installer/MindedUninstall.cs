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
                // Kill running Minded processes
                foreach (Process p in Process.GetProcessesByName("Minded"))
                {
                    try { p.Kill(); } catch { }
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
    }
}
