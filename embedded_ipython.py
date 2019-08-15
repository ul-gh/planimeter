# -*- coding: utf-8 -*-
"""Embedded Ipython Kernel and Client for Qt Application

Inspired by: https://github.com/jupyter/qtconsole/issues/197

TODO: For a non-blocking integrated widget, have a look also at
https://stackoverflow.com/questions/11513132/embedding-ipython-qt-console-in-a-pyqt-application

TODO2: Also check qtconsole.inprocess

2019-08-15 Ulrich Lukas
"""
import sys
import os
import errno
import time
import tempfile
import multiprocessing

from ipykernel.embed import embed_kernel
# IPKernelApp instance already has a blocking_client factory, thus not needed:
# from ipykernel.connect import jupyter_client
from ipykernel.kernelapp import IPKernelApp

from qtconsole.client import QtKernelClient
from qtconsole.qt import QtGui, QtCore
from qtconsole.rich_jupyter_widget import RichJupyterWidget

class EmbeddedIPythonKernel(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.conn_filename = os.path.join(
            tempfile.gettempdir(),
            f"connection-{os.getpid():d}.json"
            )
        # ?? New process must not be forked as it would then inherit some
        # QApplication state? Idk but this works:
        multiprocessing.set_start_method("spawn")
        self.console_process = None


    def start_ipython_kernel(self, namespace, **kwargs):
        try:
            # Create IPython kernel app which allows creating a connected
            # client etc.
            self.ipy_app = IPKernelApp.instance(
                connection_file=self.conn_filename,
                **kwargs,
                )
            self.ipy_app.initialize([])

            # Create a connected kernel client for shutting down the kernel
            # and possibly other uses
            self.client = self.ipy_app.blocking_client()

            # Alternative way of creating a client:
            # self.client2 = client = jupyter_client.BlockingKernelClient()
            # self.client2 = client = jupyter_client.KernelClient()
            # self.client2.load_connection_file(self.conn_filename)
            # self.client2.start_channels()

            # This is a blocking call starting the IPython kernel and exporting
            # the given namespace into the interactive context.
            # This kernel uses the already initialised ipykernel app.
            embed_kernel(local_ns=namespace)
        finally:
            try:
                os.remove(self.conn_filename)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    raise

        # When the kernel is terminated, close possibly running client process
        if self.console_process is not None:
            self.console_process.join()

    # Allow terminating the kernel process as a last step when exiting the
    # main Qt application
    @QtCore.pyqtSlot()
    def shutdown(self):
        self.client.shutdown()


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

        kernel_client = QtKernelClient(connection_file=conn_filename)
        kernel_client.load_connection_file()
        kernel_client.start_channels()

        def exit():
            kernel_client.shutdown()
            kernel_client.stop_channels()
            app.exit()

        ipython_widget = RichJupyterWidget()
        ipython_widget.kernel_client = kernel_client
        ipython_widget.exit_requested.connect(exit)
        ipython_widget.show()

        app.exec_()

