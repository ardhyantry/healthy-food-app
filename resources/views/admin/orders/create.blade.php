@extends('layouts.adminSB')
@section('title', 'Quickbite - Create Order')

@section('content')
    <div class="container">
        <h2>Tambah Order</h2>

        @if($errors->any())
            <div class="alert alert-danger">
                <ul>@foreach($errors->all() as $error) <li>{{ $error }}</li> @endforeach</ul>
            </div>
        @endif

        <form action="{{ route('admin.order.store') }}" method="POST">
            @csrf

            <div class="mb-3">
                <label>Transaksi</label>
                <select name="transactions_id" class="form-control" required>
                    @foreach($transactions as $transaction)
                        <option value="{{ $transaction->id }}">{{ $transaction->id }}</option>
                    @endforeach
                </select>
            </div>

            <div class="mb-3">
                <label>Menu</label>
                <select name="menus_id" class="form-control" required>
                    @foreach($menus as $menu)
                        <option value="{{ $menu->id }}">{{ $menu->name }}</option>
                    @endforeach
                </select>
            </div>

            <div class="mb-3">
                <label for="portion_size" class="form-label">Porsi</label>
                <select name="portion_size" id="portion_size" class="form-control" required>
                    <option value="" disabled selected>Pilih Porsi</option>
                    <option value="small">Small</option>
                    <option value="medium">Medium</option>
                    <option value="large">Large</option>
                </select>
            </div>


            <div class="mb-3">
                <label>Jumlah</label>
                <input type="number" name="quantity" class="form-control" required>
            </div>

            <div class="mb-3">
                <label>Total Harga</label>
                <input type="number" name="total" class="form-control" required>
            </div>

            <div class="mb-3">
                <label>Catatan</label>
                <textarea name="notes" class="form-control"></textarea>
            </div>

            <button type="submit" class="btn btn-primary">Simpan</button>
            <a href="{{ route('admin.order.index') }}" class="btn btn-secondary">Kembali</a>
        </form>
    </div>
@endsection