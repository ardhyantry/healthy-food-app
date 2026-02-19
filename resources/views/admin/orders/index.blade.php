@extends('layouts.adminSB')
@section('title', 'Quickbite - Orders')

@section('content')
    <div class="container">
        <h2>Daftar Order</h2>
        <a href="{{ route('admin.order.create') }}" class="btn btn-success mb-3">Tambah Order</a>

        @if(session('success'))
            <div class="alert alert-success">{{ session('success') }}</div>
        @endif

        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>User</th>
                    <th>Transaksi</th>
                    <th>Menu</th>
                    <th>Porsi</th>
                    <th>Jumlah</th>
                    <th>Total</th>
                    <th>Catatan</th>
                    <th>Aksi</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($orders as $order)
                    <tr><td>{{ $order->id }}</td>
                    <td>{{ $order->transaction->user->name ?? 'Tidak ada user' }}</td>
                    <td>{{ $order->transaction->id }}</td>
                    <td>{{ $order->menu->name }}</td>
                    <td>{{ $order->portion_size }}</td>
                    <td>{{ $order->quantity }}</td>
                    <td>{{ number_format($order->total, 0, ',', '.') }}</td>
                    <td>{{ $order->notes }}</td>
                    <td>
                        <a href="{{ route('admin.order.show', $order->id) }}" class="btn btn-info">View</a>
                        <a href="{{ route('admin.order.edit', $order->id) }}" class="btn btn-warning">Edit</a>
                        <button class="btn btn-danger" data-bs-toggle="modal" data-bs-target="#deleteModal{{ $order->id }}">
                            Delete
                        </button>
                    </td>
                </tr>
                <div class="modal fade" id="deleteModal{{ $order->id }}" tabindex="-1"
                    aria-labelledby="deleteModalLabel{{ $order->id }}" aria-hidden="true">
                    <div class="modal-dialog">
                        <div class="modal-content">
                                    <div class="modal-header bg-danger text-white">
                                        <h5 class="modal-title" id="deleteModalLabel{{ $order->id }}">Konfirmasi Hapus</h5>
                                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Tutup"></button>
                                    </div>
                                    <div class="modal-body">
                                        Apakah kamu yakin ingin menghapus Order <strong>{{ $order->menu->name }} di transaksi ke
                                            {{$order->transaction->id}}</strong>?
                                    </div>
                                    <div class="modal-footer">
                                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Batal</button>

                                        <form action="{{ route('admin.order.destroy', $order->id) }}" method="POST">
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