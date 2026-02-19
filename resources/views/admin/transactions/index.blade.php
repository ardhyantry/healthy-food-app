@extends('layouts.adminSB')

@section('title', 'Quickbite - Transactions')

@section('content')
    <div class="container">
        <h2>Daftar Transaksi</h2>
        <a href="{{ route('admin.transactions.create') }}" class="btn btn-success mb-3">Tambah Transaksi</a>

        @if (session('success'))
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                {{ session('success') }}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Tutup"></button>
            </div>
        @endif

        <table class="table table-bordered table-striped">
            <thead class="table-light">
                <tr>
                    <th>ID</th>
                    <th>Subtotal</th>
                    <th>Diskon</th>
                    <th>Total</th>
                    <th>Tipe Order</th>
                    <th>Tipe Pembayaran</th>
                    <th>Status</th>
                    <th>User ID</th>
                    <th>Dibuat</th>
                    <th>Diubah</th>
                    <th>Aksi</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($transactions as $tx)
                    <tr>
                        <td>{{ $tx->id }}</td>
                        <td>{{ number_format($tx->subtotal, 0, ',', '.') }}</td>
                        <td>{{ number_format($tx->discount, 0, ',', '.') }}</td>
                        <td>{{ number_format($tx->total, 0, ',', '.') }}</td>
                        <td>{{ ucfirst($tx->order_type) }}</td>
                        <td>{{ strtoupper($tx->payment_type) }}</td>
                        <td>{{ $tx->status }}</td>
                        <td>{{ $tx->users_id }}</td>
                        <td>{{ $tx->created_at }}</td>
                        <td>{{ $tx->updated_at }}</td>
                        <td>
                            <a href="{{ route('admin.transactions.show', $tx->id) }}" class="btn btn-info btn-sm">View</a>
                            <a href="{{ route('admin.transactions.edit', $tx->id) }}" class="btn btn-warning btn-sm">Edit</a>
                            <button class="btn btn-danger btn-sm" data-bs-toggle="modal"
                                data-bs-target="#deleteModal{{ $tx->id }}">
                                Delete
                            </button>
                        </td>
                    </tr>

                    <!-- Delete Modal -->
                    <div class="modal fade" id="deleteModal{{ $tx->id }}" tabindex="-1"
                        aria-labelledby="deleteModalLabel{{ $tx->id }}" aria-hidden="true">
                        <div class="modal-dialog">
                            <div class="modal-content">
                                <div class="modal-header bg-danger text-white">
                                    <h5 class="modal-title" id="deleteModalLabel{{ $tx->id }}">Konfirmasi Hapus</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Tutup"></button>
                                </div>
                                <div class="modal-body">
                                    Apakah kamu yakin ingin menghapus transaksi <strong>ID #{{ $tx->id }}</strong> dengan total
                                    <strong>Rp {{ number_format($tx->total, 0, ',', '.') }}</strong>?
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Batal</button>

                                    <form action="{{ route('admin.transactions.destroy', $tx->id) }}" method="POST">
                                        @csrf
                                        @method('DELETE')
                                        <button type="submit" class="btn btn-danger">Hapus</button>
                                    </form>
                                </div>
                            </div>
                        </div>
                    </div>
                @endforeach
            </tbody>
        </table>
    </div>
@endsection