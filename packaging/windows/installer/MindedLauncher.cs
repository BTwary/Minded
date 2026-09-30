using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

namespace Minded.Desktop
{
    /// <summary>
    /// Lightweight bootstrap launcher for MindEd AA-OS.
    /// Launches the self-contained bundled executable (MindedAAOS.exe).
    /// Does not require or search for system Python or Node installations.
    /// </summary>
    public static class LauncherProgram
    {
        private const string ApplicationTitle = "MindEd AA-OS";

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            string exePath = ResolveBundledExecutable();
            if (string.IsNullOrEmpty(exePath) || !File.Exists(exePath))
            {
                MessageBox.Show(
                    "MindEd AA-OS executable could not be found.\n\n" +
                    "Please verify the installation or run the installer to restore application files.",
                    ApplicationTitle,
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            try
            {
                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = exePath,
                    Arguments = string.Join(" ", args),
                    WorkingDirectory = Path.GetDirectoryName(exePath),
                    UseShellExecute = true
                };

                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Failed to launch MindEd AA-OS:\n\n" + ex.Message,
                    ApplicationTitle,
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }

        private static string ResolveBundledExecutable()
        {
            string appDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            if (string.IsNullOrEmpty(appDir)) return null;

            string[] candidates = {
                Path.Combine(appDir, "MindedAAOS.exe"),
                Path.Combine(appDir, "MindEd_AAOS.exe"),
                Path.Combine(appDir, "MindedAAOS", "MindedAAOS.exe"),
                Path.Combine(appDir, "bundle", "MindedAAOS", "MindedAAOS.exe"),
                Path.Combine(appDir, "bin", "MindedAAOS.exe")
            };

            foreach (string candidate in candidates)
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }
    }
}
