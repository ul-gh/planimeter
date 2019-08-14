import sys
import os
import errno
import time
import tempfile
import multiprocessing

import IPython

from qtconsole.client import QtKernelClient
from qtconsole.qt import QtGui, QtCore
from qtconsole.rich_jupyter_widget import RichJupyterWidget
# from PyQt5.QtWidgets import QApplication

class EmbeddedIPythonKernel(QtCore.QObject):
    quit_application = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.conn_filename = os.path.join(
            tempfile.gettempdir(),
            f"connection-{os.getpid():d}.json"
            )
        # New process must not be forked as it would then inherit some
        # QApplication state
        multiprocessing.set_start_method("spawn")
        self.console_process = None


    def start_ipython_kernel(self, namespace, **kwargs):
        try:
            # This is a blocking call
            IPython.embed_kernel(
                local_ns=namespace,
                connection_file=self.conn_filename,
                **kwargs,
                )
        finally:
            try:
                os.remove(self.conn_filename)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise
        # When the kernel is terminated, close a possibly running client
        # process and terminate application
        if self.console_process is not None:
            self.console_process.join()
        self.quit_application.emit()


    def launch_jupyter_console_process(self):
        self.console_process = multiprocessing.Process(
            target=self._run_embedded_qtconsole,
            args=(self.conn_filename,),
            )
        self.console_process.start()

    @staticmethod
    def _run_embedded_qtconsole(conn_filename):
        # This is launched as a new process.
        #
        # Wait max. ten seconds for the IPython kernel to be up and running,
        # which is done by checking for presence of the connection file
        # containing the kernels Zero Message Queue sockets and credentials.
        #
        # Then, start a new QApplication running the Jupyter Console client
        # widget connected to the kernel via the connection file.
        # 
        for i in range(100):
            try:
                st = os.stat(conn_filename)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    # No such file exception is ignored, all others re-raised
                    raise
            else:
                if st.st_size > 0:
                    # OK, connection file found, kernel seems to be running
                    break
            time.sleep(0.1)

        app = QtGui.QApplication(["Plot Workbench Console"])
        # app = QApplication(["Plot Workbench Console"])

        kernel_client = QtKernelClient(connection_file=conn_filename)
        kernel_client.load_connection_file()
        kernel_client.start_channels()

        def exit():
            # FIXME: tell the kernel to shutdown
            kernel_client.shutdown()
            kernel_client.stop_channels()
            app.exit()

        ipython_widget = RichJupyterWidget()
        ipython_widget.kernel_client = kernel_client
        ipython_widget.exit_requested.connect(exit)
        ipython_widget.show()

        app.exec_()

