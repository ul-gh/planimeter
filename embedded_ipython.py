import sys
import os
import errno
import time
import tempfile
import multiprocessing

import IPython

from qtconsole.client import QtKernelClient
from qtconsole.qt import QtGui
from qtconsole.rich_jupyter_widget import RichJupyterWidget

class EmbeddedIPythonKernel():
    def __init__(self, export_namespace, **kwargs):
        self.conn_filename = os.path.join(
            tempfile.gettempdir(),
            f"connection-{os.getpid():d}.json"
            )
        self.export_namespace = export_namespace
        self.kwargs = kwargs

    def embed_jupyter_console(self):
        try:
            console_process = self._launch_console_process()
            # This is a blocking call
            self._embed_kernel()
            # When the kernel has died, terminate the console
            console_process.join()
        finally:
            try:
                os.remove(conn_filename)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise


    def _embed_kernel(self):
        IPython.embed_kernel(
            local_ns=self.export_namespace,
            connection_file=self.conn_filename,
            **self.kwargs,
            )

    def _launch_console_process(self):
        console_process = multiprocessing.Process(
            target=self._run_embedded_qtconsole,
            args=(self.conn_filename,),
            )
        console_process.start()
        return console_process

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

        app = QtGui.QApplication([])

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

