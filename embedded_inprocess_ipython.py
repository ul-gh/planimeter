#from qtconsole.qt import QtGui
import sys
from PyQt5.Qt import QApplication
from PyQt5.QtWidgets import QVBoxLayout, QLineEdit
from qtconsole.client import QtKernelClient
from qtconsole.rich_jupyter_widget import RichJupyterWidget
#from qtconsole.inprocess import QtInProcessKernelManager


class ConsoleWidget(RichJupyterWidget):
    def __init__(self, customBanner=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if customBanner is not None:
            self.banner = customBanner

        self.font_size = 6
#        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
#        kernel_manager.start_kernel(show_banner=False)
#        kernel_manager.kernel.gui = 'qt'
#        self.kernel_client = kernel_client = self._kernel_manager.client()
#        filename = "/tmp/connfile.json"
        filename = "/tmp/connfile"
        kernel_client = QtKernelClient(connection_file=filename)
        kernel_client.load_connection_file()
        kernel_client.start_channels()
        self.kernel_client = kernel_client

        def stop():
            kernel_client.shutdown()
            kernel_client.stop_channels()
#            kernel_manager.shutdown_kernel()
            #QApplication.instance().exit()

        self.exit_requested.connect(stop)


    def push_vars(self, variableDict):
        """
        Given a dictionary containing name / value pairs, push those variables
        to the Jupyter console widget
        """
#        self.kernel_manager.kernel.shell.push(variableDict)
        return

    def clear(self):
        """
        Clears the terminal
        """
        self._control.clear()

        # self.kernel_manager

    def print_text(self, text):
        """
        Prints some plain text to the console
        """
        self._append_plain_text(text)

    def execute_command(self, command):
        """
        Execute a command in the frame of the console widget
        """
        self._execute(command, False)

